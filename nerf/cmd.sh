 
 cd /media/vrlab/rabbit/print3dingp/instant-ngp
 
conda activate ingp
 ./instant-ngp data/nerf/fox

 --load_snapshot data/nerf/fox/output/base.ingp
 
 --load_snapshot data/nerf/fox/output/base.ingp --gui
 --load_snapshot data/nerf/fox/output/base.ingp --save_volume data/nerf/fox/output/volume/
 
 conda activate torch-ngp

--workspace 👑Train/tmp

python scripts/colmap2nerf.py --images "/media/vrlab/rabbit/print3dingp/print_ngp/data/nerf_llff_data/trex/images" --run_colmap
# isoOrAniso=iso

datasetPath=data/fox  && animal=$(basename "$datasetPath") && bbox=(-0.74 -0.76 -0.22 0.34 0.64 0.78)
datasetPath=data/new/duck && animal=$(basename "$datasetPath") && bbox=(-0.08 -0.20 -0.10 0.13 0.09 0.16)
datasetPath=data/artemis/panda && animal=$(basename "$datasetPath")  && bbox=(-0.19 -0.12 -0.24 0.27 0.09 0.41)
datasetPath=/media/vrlab/rabbit/print3dingp/print_ngp/data/artemis/panda  && animal=$(basename "$datasetPath")  && bbox=(-0.15 -0.25 -0.15 0.35 0.25 0.45)
datasetPath=/media/vrlab/rabbit/print3dingp/print_ngp/data/nerf_synthetic/lego  && animal=$(basename "$datasetPath")  && bbox=(-0.39 -0.13 -0.26 0.39 0.39 0.27)

experiment=🐲Anisotropy && datasetPath=/media/vrlab/rabbit/print3dingp/print_ngp/data/nerf_llff_data/fern  && animal=$(basename "$datasetPath")  && bbox=(-0.82 -0.46 -0.69 0.66 0.46 0.78) && isoOrAniso=aniso && density=-1
experiment=🐲Anisotropy && datasetPath=/media/vrlab/rabbit/print3dingp/print_ngp/data/nerf_llff_data/trex  && animal=$(basename "$datasetPath")  && bbox=() && isoOrAniso=aniso && density=-1

experiment=🐲isotropy

axisDir=

python main_nerf.py $datasetPath --workspace ${experiment}/${animal}_d$density -O --iters 10000 --density_max_scale $density --isoOrAniso $isoOrAniso &&  python main_nerf.py $datasetPath --workspace ${experiment}/${animal}_d$density --save_volume --volume_bbox $bbox --printing_height 20 --edit_axis 0 1 2 --density_max_scale $density --isoOrAniso $isoOrAniso

python main_nerf.py $datasetPath --workspace ${experiment}/${animal}_d$density --test --gui  --density_max_scale $density --isoOrAniso $isoOrAniso # 查看bbox

volume_folder=$(ls -d ${experiment}/${animal}_d${density}/volume/ngp_* | head -n 1)
python volume2print_blur.py --input_folder $volume_folder --output_folder $volume_folder && python preview_print_volume_blend.py --input_folder $volume_folder


python preview_print_volume_blend.py --input_folder 🚯bin


datasetPath=/media/vrlab/rabbit/print3dingp/print_ngp_lyf/data/new/duck && animal=$(basename "$datasetPath") && bbox=-0.08 -0.20 -0.10 0.13 0.09 0.16


