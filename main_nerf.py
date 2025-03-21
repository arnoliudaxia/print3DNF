import torch
import argparse

from nerf.provider import NeRFDataset
import os
if os.environ.get('DISPLAY', '') != '':
    from nerf.gui import NeRFGUI
from nerf.utils import *

from functools import partial
from loss import huber_loss

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### save volume options
    parser.add_argument('--save_volume', action='store_true', help="save volume")
    parser.add_argument('--volume_bbox',type=float, nargs=6, default=[-1., -1., -1., 1., 1., 1.], help="volume scale in nerf-space (use --test --gui to get a perfect number)")
    parser.add_argument('--voxel_size',type=float, nargs=3, default=[0.0846666, 0.042333, 0.014], help="(mm) for 3D print")
    parser.add_argument('--printing_height',type=float, default=10.0, help="(mm) the height of 3D printing object, width and length will be calculated automatically")
    parser.add_argument('--edit_axis',type=int, nargs=3, default=[0, 1, 2], help="for temporary use, swap or flip axes")

    ### training options
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")
    
    
    ###my
    parser.add_argument('--density_max_scale', type=int, default=-1, help="Deprecated, no longer limits density during training")
    parser.add_argument('--isoOrAniso', type=str, help="iso means no colorNet, vice versa ", default="aniso")
    parser.add_argument('--noMask', action='store_false')
    parser.add_argument('--numViewsMean', type=int,default=20, help="How many views to adapt when extract volume")
    

    opt = parser.parse_args()

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True
    
    if opt.patch_size > 1:
        opt.error_map = False # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."


    if opt.ff:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --ff"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
        from nerf.network_tcnn import NeRFNetwork
    else:
        if opt.isoOrAniso=="iso":
            from nerf.network import NeRFNetwork
        elif opt.isoOrAniso=="aniso":
            from nerf.network_aniso import NeRFNetwork
        else:
            raise()

    print(opt)
    
    seed_everything(opt.seed)

    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=True,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
        density_max_scale=opt.density_max_scale,
    )
    
    
    print(model)

    criterion = torch.nn.MSELoss(reduction='none')
    #criterion = partial(huber_loss, reduction='none')
    #criterion = torch.nn.HuberLoss(reduction='none', beta=0.1) # only available after torch 1.10 ?

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.test:
        
        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            testset = NeRFDataset(opt, device=device, type='test')
            test_loader = testset.dataloader()

            if test_loader.has_gt:
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
    
            trainer.test(test_loader, write_video=False) #! test and save video
            
    elif opt.save_volume:
        
        # 采样一些train里的direction
        train_loader = NeRFDataset(opt, device="cpu", type='train').dataloader()
        # Step 1: 收集所有的rays_d
        rays_directions = []
        for batch in train_loader:
            # batch['rays_d'].shape #torch.Size([1, 4096, 3])
            rays_directions.append(batch['rays_d'][0])  # 将每个batch的rays_d收集起来
        # 将列表转为一个大tensor
        rays_directions = torch.cat(rays_directions, dim=0)  # 现在是一个[总样本数, 4096, 3]的tensor
        rays_directions = rays_directions.view(-1, 3)  # 将维度展平为[n, 3]形式
        # 随机采样20个样本
        num_samples = opt.numViewsMean
        random_indices = torch.randint(0, rays_directions.shape[0], (num_samples,))

        # 使用这些索引来获取样本
        sampled_rays_directions = rays_directions[random_indices]
        # sampled_rays_directions = torch.tensor([[-8.0199050e-02, -6.9577720e-03, 9.675459e-01]]) #! hack for fern
        
        
        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)
        
        volume_bbox = [opt.volume_bbox[axis] for axis in opt.edit_axis[0:3]] + [opt.volume_bbox[axis + 3] for axis in opt.edit_axis[0:3]]
        lx, ly, lz = volume_bbox[3] - volume_bbox[0], volume_bbox[4] - volume_bbox[1], volume_bbox[5] - volume_bbox[2]
        
        printing_size = np.array([lx/lz*opt.printing_height, ly/lz*opt.printing_height, opt.printing_height])
        volume_resolution = np.floor(printing_size / np.array(opt.voxel_size)).astype(np.int32)
        
        print(f"==> Printing object size is: {printing_size} (mm)")
        print(f"==> Volume resolution is: {volume_resolution}")
        trainer.save_volume(volume_bbox, resolution=list(volume_resolution), edit_axis=opt.edit_axis, sampleDirs=sampled_rays_directions, shouldMask=opt.noMask)
    
    else:

        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        train_loader = NeRFDataset(opt, device=device, type='train').dataloader()

        # decay to 0.1 * init_lr at last iter step
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=50)

        if opt.gui:
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()
        
        else:
            valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()

            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            trainer.train(train_loader, valid_loader, max_epoch)

            # also test
            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()
            
            if test_loader.has_gt:
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            
            trainer.test(test_loader, write_video=True) # test and save video
            
            trainer.save_mesh(resolution=256, threshold=10)