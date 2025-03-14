 
 cd /media/vrlab/rabbit/print3dingp/instant-ngp
 
conda activate ingp
 ./instant-ngp data/nerf/fox

 --load_snapshot data/nerf/fox/output/base.ingp
 
 --load_snapshot data/nerf/fox/output/base.ingp --gui
 --load_snapshot data/nerf/fox/output/base.ingp --save_volume data/nerf/fox/output/volume/
 
 conda activate torch-ngp

--workspace 👑Train/tmp

python scripts/colmap2nerf.py --images "/media/vrlab/rabbit/print3dingp/print_ngp/data/nerf_llff_data/trex/images" --run_colmap

experiment=🐲Anisotropy && datasetPath=data/new/duck && animal=$(basename "$datasetPath") && bbox=(-0.08 -0.20 -0.10 0.13 0.09 0.16) && isoOrAniso=aniso && density=-1
experiment=🐲Anisotropy && datasetPath=/media/vrlab/rabbit/print3dingp/print_ngp/data/nerf_llff_data/fern  && animal=$(basename "$datasetPath")  && bbox=(-0.82 -0.46 -0.69 0.66 0.46 0.78) && isoOrAniso=aniso && density=-1
experiment=🐲Anisotropy && datasetPath=/media/vrlab/rabbit/print3dingp/print_ngp/data/nerf_llff_data/trex  && animal=$(basename "$datasetPath")  && bbox=() && isoOrAniso=aniso && density=-1



axisDir=
#  ______
# < SOTA >
#  ------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||

isoOrAniso=aniso && density=-1 # 务必执行

experiment=🃏CUDA && datasetPath=/media/vrlab/rabbit/print3dingp/print_ngp/data/nerf_synthetic/lego  && animal=$(basename "$datasetPath")  && bbox=(-0.39 -0.13 -0.26 0.39 0.39 0.27) && axis=(0 1 2)
experiment=🃏CUDA && datasetPath=data/fox  && animal=$(basename "$datasetPath") && bbox=(-0.74 -0.76 -0.22 0.34 0.64 0.87) && axis=(0 1 2)
experiment=🃏CUDA &&datasetPath=/media/vrlab/rabbit/print3dingp/print_ngp/data/new_panda  && animal=$(basename "$datasetPath")  && bbox=(-0.15 -0.25 -0.15 0.35 0.25 0.45) && axis=(2 1 0)
experiment=🃏CUDA &&datasetPath=/media/vrlab/rabbit/print3dingp/print_ngp/data/nerf_llff_data/fern  && animal=$(basename "$datasetPath")  && bbox=(-0.90 -0.46 -0.64 0.96 0.51 0.58) && axis=(0 1 2)
experiment=🃏CUDA &&datasetPath=/media/vrlab/rabbit/print3dingp/print_ngp/data/nerf_llff_data/trex  && animal=$(basename "$datasetPath")  && bbox=(-0.69 -0.77 -0.49 1.24 0.34 0.39)  && axis=(0 1 2)
experiment=🃏CUDA &&datasetPath=data/artemis/duck-best && animal=$(basename "$datasetPath")  && bbox=(-0.05 -0.03 -0.04 0.05 0.05 0.04)  && axis=(0 1 2)




python main_nerf.py $datasetPath --workspace ${experiment}/${animal}_d$density -O --iters 18000 $extraCMD
python  main_nerf.py $datasetPath --workspace ${experiment}/${animal}_d$density -O --iters 8000   --cuda_ray --fp16 --num_rays 8888 --gui # 训练nerf

# 需要先注释 _run = self.run_cuda 
python main_nerf.py $datasetPath --workspace ${experiment}/${animal}_d$density --test --gui  # 查看bbox

#region 去掉mask
# experiment=🐶NOMASK && datasetPath=/media/vrlab/rabbit/print3dingp/print_ngp/data/nerf_synthetic/ficus && animal=$(basename "$datasetPath") && axis=(0 1 2) && extraCMD=(--noMask --fp16)  && bbox=(-0.30 -0.40 -0.18 0.21 0.48 0.26)
# experiment=🐶NOMASK && density=-1 && printheight=20 && datasetPath=data/nerf_synthetic/lego  && animal=$(basename "$datasetPath")  && bbox=(-0.39 -0.13 -0.26 0.39 0.39 0.27) && axis=(0 1 2) && extraCMD=(--noMask --fp16)
# experiment=🐶NOMASK &&datasetPath=data/artemis/duck-best && animal=$(basename "$datasetPath")  && bbox=(-0.05 -0.03 -0.04 0.05 0.05 0.04)  && axis=(0 1 2) && extraCMD=(--noMask --fp16)
# experiment=🐶NOMASK && datasetPath=data/fox  && animal=$(basename "$datasetPath") && bbox=(-0.74 -0.76 -0.22 0.34 0.64 0.78) && axis=(0 1 2)
# experiment=🐶NOMASK &&datasetPath=/media/vrlab/rabbit/print3dingp/print_ngp/data/nerf_llff_data/fern  && animal=$(basename "$datasetPath")  && bbox=(-0.90 -0.46 -0.64 0.96 0.51 0.58) && axis=(0 1 2)
# experiment=🐶NOMASK &&datasetPath=/media/vrlab/rabbit/print3dingp/print_ngp/data/nerf_llff_data/trex  && animal=$(basename "$datasetPath")  && bbox=(-0.69 -0.77 -0.49 1.24 0.34 0.39)  && axis=(0 1 2)

# experiment=🐶NOMASK && density=-1 && printheight=20 && datasetPath=data/nerf_synthetic/ficus && animal=$(basename "$datasetPath") && axis=(0 1 2) && extraCMD=(--noMask --fp16)  && bbox=(-0.30 -0.40 -0.18 0.21 0.48 0.26)
# experiment=🐶NOMASK && density=-1 && printheight=20 && datasetPath=data/nerf_synthetic/ficus && animal=$(basename "$datasetPath") && axis=(0 1 2) && extraCMD=(--noMask --fp16)  && bbox=(-0.30 -0.38 -0.13 0.21 0.46 0.21)
# experiment=🐶NOMASK && density=-1 && printheight=20 && datasetPath=data/nerf_synthetic/ficus && animal=$(basename "$datasetPath") && axis=(0 1 2) && extraCMD=(--noMask --fp16)  && bbox=(-0.30 -0.38 -0.5 0.21 0.46 0.10)
# # experiment=🐶NOMASK && density=-1 && printheight=20 && datasetPath=data/nerf_synthetic/ficus && animal=$(basename "$datasetPath") && axis=(0 1 2) && extraCMD=(--noMask --fp16)  && bbox=(-0.25 -0.40 -0.18 0.18 0.48 0.26)

# experiment=🐶NOMASK && density=-1 && printheight=20 && datasetPath=data/artemis/panda/new_panda  && animal=$(basename "$datasetPath")  && bbox=(-0.13 -0.20 -0.10 0.31 0.19 0.38) && axis=(2 1 0) && extraCMD=(--noMask --numViewsMean 1  --cuda_ray --fp16 --num_rays 4096)

#endregion


# 高分辨率 height 40
# experiment=🔎highResolution && density=scale10 && printheight=40 && datasetPath=data/artemis/duck-best && animal=$(basename "$datasetPath")  && bbox=(-1.36 -0.73 -1.04 1.34 1.51 0.94)  && axis=(0 1 2) && extraCMD=(--noMask --fp16 --num_rays 8888 --scale 10 --numViewsMean 3)
# experiment=🔎highResolution && density=-1 && printheight=40 && datasetPath=data/nerf_llff_data/fern/fern && animal=$(basename "$datasetPath")  && bbox=(-0.90 -0.46 -0.64 0.96 0.51 0.58) && axis=(0 1 2)&& extraCMD=(--noMask --fp16)
# experiment=🔎highResolution && density=-1 && printheight=20 && datasetPath=data/trex  && animal=$(basename "$datasetPath")  && bbox=(-0.69 -0.77 -0.49 1.24 0.34 0.39)  && axis=(0 1 2) && extraCMD=(--noMask --fp16)
# experiment=🔎highResolution && density=-1 && printheight=40 && datasetPath=data/nerf_synthetic/ficus && animal=$(basename "$datasetPath") && axis=(0 1 2) && extraCMD=(--noMask --fp16)  && bbox=(-0.30 -0.38 -0.5 0.21 0.46 0.10)
# experiment=🔎highResolution && density=-1 && printheight=40 && datasetPath=data/artemis/panda/new_panda  && animal=$(basename "$datasetPath")  && bbox=(-0.20 -0.62 -0.57 1.14 0.53 1.33) && axis=(2 1 0) && extraCMD=(--noMask --numViewsMean 3 --scale 1.2 --cuda_ray --fp16 --num_rays 4096)

# experiment=🔎highResolution && density=-1 && datasetPath=data/garden  && animal=$(basename "$datasetPath") && extraCMD=(--noMask --fp16  --num_rays 8888 --bg_radius 1.1) 


# experiment=🔎highResolution &&  density=-1 && printheight=40 && datasetPath=data/fox  && animal=$(basename "$datasetPath") && bbox=(-0.74 -0.76 -0.22 0.34 0.64 0.78) && axis=(0 1 2) && extraCMD=(--noMask --fp16 --numViewsMean 3)  

# experiment=💾HDDSnake &&  density=-1 && printheight=40 && datasetPath=data/nerf_synthetic/lego  && animal=$(basename "$datasetPath")  && bbox=(-0.39 -0.13 -0.26 0.39 0.39 0.27) && axis=(0 1 2) && extraCMD=(--noMask --fp16)


python main_nerf.py $datasetPath --workspace ${experiment}/${animal}_d$density --save_volume --volume_bbox $bbox --printing_height $printheight --edit_axis $axis $extraCMD # ⭐

volume_folder=$(ls -d ${experiment}/${animal}_d${density}/volume/ngp_* | head -n 1) && python /media/vrlab/rabbit/print3dingp/print_volume/preview_volume.py --input_folder ${volume_folder}/array  && python /media/vrlab/rabbit/print3dingp/print_volume/volume2print_batch.py --input_folder ${volume_folder} --output_folder ${volume_folder} && python /media/vrlab/rabbit/print3dingp/print_volume/preview_volume.py --input_folder ${volume_folder}/pred_rgbd --savePrefix printReady

 volume_folder=$(ls -d ${experiment}/${animal}_d${density}/volume/ngp_* | head -n 1) 

## mask 归来

# experiment=💾HDDSnake && density=scale10 && printheight=20 && maskProportion=0.0 && datasetPath=data/artemis/duck-best && animal=$(basename "$datasetPath")  && bbox=(-1.36 -0.73 -1.04 1.34 1.51 0.94)  && axis=(0 1 2) && extraCMD=( --fp16 --num_rays 8888 --scale 10 --numViewsMean 3)

# experiment=🐶MASK && density=scale10 && printheight=30 && maskProportion=0.66 && datasetPath=data/artemis/duck-best && animal=$(basename "$datasetPath")  && bbox=(-1.36 -0.73 -1.04 1.34 1.51 0.94)  && axis=(0 1 2) && extraCMD=( --fp16 --num_rays 8888 --scale 10 --numViewsMean 6) # 需要多mask
# experiment=🐶MASK && density=-1 && printheight=20 && maskProportion=0.0 && datasetPath=data/nerf_llff_data/fern/fern && animal=$(basename "$datasetPath")  && bbox=(-0.90 -0.46 -0.64 0.96 0.51 0.58) && axis=(0 1 2)&& extraCMD=( --fp16)
# experiment=🐶MASK && density=-1 && printheight=20 && maskProportion=0.0 && datasetPath=data/artemis/panda/new_panda  && animal=$(basename "$datasetPath")  && bbox=(-0.20 -0.62 -0.57 1.14 0.53 1.33) && axis=(2 1 0) && extraCMD=( --numViewsMean 3 --scale 1.2 --cuda_ray --fp16 --num_rays 4096)

# experiment=🐶MASK && density=-1 && printheight=20 && maskProportion=0.0 && datasetPath=data/lego_diffuse  && animal=$(basename "$datasetPath")  && bbox=(-0.39 -0.13 -0.26 0.39 0.39 0.27) && axis=(0 1 2) && extraCMD=(--fp16 --numViewsMean 10)

# experiment=💾HDDSnake/🐶MASK && density=-1 && printheight=30 && maskProportion=0.0 && datasetPath=data/lego_diffuse  && animal=$(basename "$datasetPath")  && bbox=(-0.39 -0.13 -0.26 0.39 0.39 0.27) && axis=(0 1 2) && extraCMD=(--fp16 --numViewsMean 10 --test)
# experiment=💾HDDSnake/🏆FINAL && density=-1 && printheight=30 && maskProportion=0.0 && datasetPath=data/lego_diffuse  && animal=$(basename "$datasetPath")  && bbox=(-0.39 -0.13 -0.26 0.39 0.39 0.27) && axis=(0 1 2) && extraCMD=(--fp16 --numViewsMean 10)
# experiment=💾HDDSnake/🏆FINAL && density=-1 && printheight=20 && maskProportion=0.0 && datasetPath=data/lego_diffuse  && animal=$(basename "$datasetPath")  && bbox=(-0.39 -0.13 -0.26 0.39 0.39 0.27) && axis=(0 1 2) && extraCMD=(--fp16 --numViewsMean 10)

experiment=💾HDDSnake/🏆FINAL && density=-1 && printheight=20 && maskProportion=0.0 && datasetPath=data/lego_diffuse  && animal=$(basename "$datasetPath")  && bbox=(-0.39 -0.13 -0.26 0.39 0.39 0.27) && axis=(0 1 2) && extraCMD=(--fp16 --numViewsMean 10 )



# volume_folder=🐶VDB-Release/typical_building_colorful_cottage&& python /media/vrlab/rabbit/print3dingp/print_volume/volume2print_batch.py --input_folder ${volume_folder} --output_folder ${volume_folder} && python /media/vrlab/rabbit/print3dingp/print_volume/preview_volume.py --input_folder ${volume_folder}/pred_rgbd --onlyOneView
# volume_folder=🐶VDB-Release/typical_building_building&& python /media/vrlab/rabbit/print3dingp/print_volume/volume2print_batch.py --input_folder ${volume_folder} --output_folder ${volume_folder} && python /media/vrlab/rabbit/print3dingp/print_volume/preview_volume.py --input_folder ${volume_folder}/pred_rgbd --onlyOneView
# volume_folder=🐶VDB-Release/typical_building_castle&& python /media/vrlab/rabbit/print3dingp/print_volume/volume2print_batch.py --input_folder ${volume_folder} --output_folder ${volume_folder} && python /media/vrlab/rabbit/print3dingp/print_volume/preview_volume.py --input_folder ${volume_folder}/pred_rgbd --onlyOneView
# volume_folder=🐶VDB-Release/typical_creature_dragon&& python /media/vrlab/rabbit/print3dingp/print_volume/volume2print_batch.py --input_folder ${volume_folder} --output_folder ${volume_folder} && python /media/vrlab/rabbit/print3dingp/print_volume/preview_volume.py --input_folder ${volume_folder}/pred_rgbd --onlyOneView
# volume_folder=🐶VDB-Release/typical_creature_furry&& python /media/vrlab/rabbit/print3dingp/print_volume/volume2print_batch.py --input_folder ${volume_folder} --output_folder ${volume_folder} && python /media/vrlab/rabbit/print3dingp/print_volume/preview_volume.py --input_folder ${volume_folder}/pred_rgbd --onlyOneView
# volume_folder=🐶VDB-Release/typical_vehicle_pirate_ship&& python /media/vrlab/rabbit/print3dingp/print_volume/volume2print_batch.py --input_folder ${volume_folder} --output_folder ${volume_folder} && python /media/vrlab/rabbit/print3dingp/print_volume/preview_volume.py --input_folder ${volume_folder}/pred_rgbd --onlyOneView



# volume_folder=🐶VDB-Release/ten1o_A_miniature__of_a_classical_Suzhou_garden__perched_on_a_f_71b11dd4-fa2a-4e3c-ae67-9ae232836644 && python /media/vrlab/rabbit/print3dingp/print_volume/volume2print_batch.py --input_folder ${volume_folder} --output_folder ${volume_folder} && python /media/vrlab/rabbit/print3dingp/print_volume/preview_volume.py --input_folder ${volume_folder}/pred_rgbd --onlyOneView
# volume_folder=🐶VDB-Release/ten1o_A_miniature_floating_Suzhou_garden__with_a_small_elegant__29ecf1fe-5a57-42c2-89ec-1b8261d22a63 && python /media/vrlab/rabbit/print3dingp/print_volume/volume2print_batch.py --input_folder ${volume_folder} --output_folder ${volume_folder} && python /media/vrlab/rabbit/print3dingp/print_volume/preview_volume.py --input_folder ${volume_folder}/pred_rgbd --onlyOneView

# experiment=💾HDDSnake/🐶MASK &&  density=-1 && printheight=30 && maskProportion=0.66 && datasetPath=data/fox  && animal=$(basename "$datasetPath") && bbox=(-0.74 -0.76 -0.22 0.34 0.64 0.84) && axis=(0 1 2) && extraCMD=( --fp16 --numViewsMean 3) 
experiment=💾HDDSnake/🐶MASK && density=-1 && printheight=30 && maskProportion=0.0 && datasetPath=data/artemis/panda/new_panda  && animal=$(basename "$datasetPath")  && bbox=(-0.20 -0.62 -0.57 1.14 0.53 1.33) && axis=(2 1 0) && extraCMD=( --numViewsMean 3 --scale 1.2 --cuda_ray --fp16 --num_rays 4096)
# experiment=💾HDDSnake/🐶MASK && density=scale10 && printheight=30 && maskProportion=0.1 && datasetPath=data/artemis/duck-best && animal=$(basename "$datasetPath")  && bbox=(-1.36 -0.73 -1.04 1.34 1.51 0.94)  && axis=(0 1 2) && extraCMD=( --fp16 --num_rays 8888 --scale 10 --numViewsMean 2) 
# experiment=💾HDDSnake/🐶MASK && density=-1 && printheight=15 && maskProportion=0.0 && datasetPath=data/nerf_synthetic/ficus && animal=$(basename "$datasetPath") && axis=(0 1 2) && extraCMD=( --fp16)  && bbox=(-0.30 -0.36 -0.25 0.21 0.46 0.10)
# experiment=💾HDDSnake/🐶MASK && density=-1 && printheight=15 && maskProportion=0.0 && datasetPath=data/nerf_synthetic/ficus && animal=$(basename "$datasetPath") && axis=(0 2 1) && extraCMD=( --fp16)  && bbox=(-0.33 -0.36 -0.27 0.33 0.46 0.27)


# experiment=💾HDDSnake/🐶MASK && density=-1 && printheight=20 && maskProportion=0.0 && datasetPath=data/nerf_llff_data/fern/fern && animal=$(basename "$datasetPath")  && bbox=(-1.11 -0.52 -1.81 0.88 0.54 0.55)&& axis=(0 1 2)&& extraCMD=( --fp16 --numViewsMean 2 --noMask) 
# experiment=💾HDDSnake/🐶MASK && density=-1 && printheight=20 && maskProportion=0.0 && datasetPath=data/nerf_llff_data/fern/fern && animal=$(basename "$datasetPath")  && bbox=(-1.11 -0.52 -0.27 0.88 0.54 0.55)&& axis=(0 1 2)&& extraCMD=( --fp16 --numViewsMean 2 --noMask) 
# experiment=💾HDDSnake/🐶MASK && density=-1 && printheight=20 && maskProportion=0.0 && datasetPath=data/nerf_llff_data/fern/fern && animal=$(basename "$datasetPath")  && bbox=(-1.11 -0.52 -1.9 0.88 0.54 0.55)&& axis=(0 1 2)&& extraCMD=( --fp16 --numViewsMean 2 --noMask) 
#  玻璃大概在z=-0.27 这里
## Runner

 && volume_folder=$(ls -d ${experiment}/${animal}_d${density}/volume/ngp_* | head -n 1)  && python /media/vrlab/rabbit/print3dingp/print_volume/volume2print_batch.py --maskProportion $maskProportion --input_folder ${volume_folder} --output_folder ${volume_folder} && python /media/vrlab/rabbit/print3dingp/print_volume/preview_volume.py --input_folder ${volume_folder}/pred_rgbd --savePrefix printReady-black

python /media/vrlab/rabbit/print3dingp/print_volume/preview_volume.py --input_folder ${volume_folder}/array --pinkBackground


 python main_nerf.py $datasetPath --workspace ${experiment}/${animal}_d$density --save_volume --volume_bbox $bbox --printing_height $printheight --edit_axis $axis $extraCMD --test
 python main_nerf.py $datasetPath --workspace ${experiment}/${animal}_d$density --volume_bbox $bbox --printing_height $printheight --edit_axis $axis --test

volume_folder=$(ls -d ${experiment}/${animal}_d${density}/volume/ngp_* | head -n 1)  &&   python /media/vrlab/rabbit/print3dingp/print_volume/preview_volume.py --input_folder ${volume_folder}/pred_rgbd --savePrefix printReady

python /media/vrlab/rabbit/print3dingp/print_volume/preview_volume.py --input_folder ${volume_folder}/array --pinkBackground

volume_folder=$(ls -d ${experiment}/${animal}_d${density}/volume${printheight}/ngp_* | head -n 1)  && python /media/vrlab/rabbit/print3dingp/print_volume/volume2print_batch.py --maskProportion $maskProportion --input_folder ${volume_folder} --output_folder ${volume_folder} && python /media/vrlab/rabbit/print3dingp/print_volume/preview_volume.py --input_folder ${volume_folder}/pred_rgbd --savePrefix printReady


#### noMask是有mask）, 现在已经修复了

### 20250206
# 把之前错误尺寸的fern修复一下
# 放弃环境，使用docker！
# docker run -it --rm --gpus all -v /home/arno/Projects/Pint3D:/workspace pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
apt-get update
apt-get install -y curl git
# bash <(curl -sSL https://linuxmirrors.cn/main.sh)
mkdir -p  /root/.cache/torch/hub/checkpoints/
ln -s /root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth 
 /root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth 
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install ninja trimesh opencv-contrib-python-headless tensorboardX numpy pandas tqdm matplotlib rich packaging scipy
pip install imageio lpips torch-ema PyMCubes pysdf dearpygui torchmetrics
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

docker run -it --rm --gpus all -v /home/arno/Projects/Pint3D:/workspace arnoliu/print3dnerf:v0

export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_HOME=$CONDA_PREFIX


printheight=40 && maskProportion=0.0 && datasetPath=../nerf_scene_data/fern && animal=$(basename "$datasetPath")  && bbox=(-1.11 -0.52 -1.9 0.88 0.54 0.55)&& axis=(0 1 2)&& extraCMD=( --fp16 --numViewsMean 2 --noMask) 
#  玻璃大概在z=-0.27 这里

python main_nerf.py $datasetPath --workspace ../print_data/${animal} --save_volume --volume_bbox ${bbox[@]} --printing_height $printheight --edit_axis  ${axis[@]} $extraCMD

volume_folder=$(ls -d ../print_data/${animal}/volume/ngp_* | head -n 1) && python ../print_volume/volume2print_batch.py --maskProportion $maskProportion --input_folder ${volume_folder} --output_folder ${volume_folder} && python ../print_volume/preview_volume.py  --input_folder ${volume_folder}/pred_rgbd --savePrefix printReady

python ../print_volume/preview_volume.py --input_folder ${volume_folder}/array --savePrefix printReady

## rsic

# copy checkpoints (start a new experiment)
rsync -av --include='*/' --include='checkpoints/***' --exclude='*' ./source_directory/ ./destination_directory/ 
# move experiment to HDD to save space
rsync -av --progress "$SOURCE/" "$DESTINATION/$(basename "$SOURCE")" 
rm -rf "$SOURCE"
ln -s "$DESTINATION/$(basename "$SOURCE")" "$SOURCE"


# whiteFix
python ../print_volume/volume2print_batch.py  --input_folder /home/arno/Projects/Pint3D/print_data/typical_creature_furry
python ../print_volume/volume2print_batch.py  --input_folder /home/arno/Projects/Pint3D/print_ngp/mylut

python ../print_volume/volume2print_batch_GradientAgg.py  --input_folder /home/arno/Projects/Pint3D/print_data/test_slice

&& python ../print_volume/preview_volume.py  --input_folder ${volume_folder}/pred_rgbd --savePrefix printReady

python ../print_volume/preview_volume.py  --input_folder /home/arno/Projects/Pint3D/print_data/typical_creature_furry/pred_rgbd --savePrefix printReady




python ../print_volume/preview_printslice.py --input_folder mylut/printImg/print/color1mm_White1mm/mode1/slice --savePrefix printReady



python ../print_volume/volume2print_batch_GradientAgg.py  --input_folder /home/arno/Projects/Pint3D/print_data/Gau/lego_ner/lego_crop_newtest --adjust_density False


python ../print_volume/volume2print_batch.py  --input_folder  /home/arno/Projects/Pint3D/print_data/Gau/sht
python ../print_volume/preview_volume.py  --input_folder /home/arno/Projects/Pint3D/print_data/Gau/sht/pred_rgbd --savePrefix printReady


