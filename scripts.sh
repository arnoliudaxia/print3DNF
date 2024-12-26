python main_nerf.py data/nerf_synthetic/lego --workspace workspace/lego --save_volume --volume_bbox -0.39 -0.13 -0.26 0.39 0.39 0.27 --printing_height 20 --edit_axis 0 2 1
python main_nerf.py data/nerf_synthetic/lego --workspace workspace/lego --test --gui

python main_nerf.py data/fox --workspace workspace/fox --save_volume --volume_bbox  -0.74 -0.76 -0.22 0.34 0.64 0.78 --printing_height 20 --edit_axis 0 1 2
python main_nerf.py data/fox --workspace workspace/fox --test --gui

python main_nerf.py data/nerf_synthetic/ficus --workspace workspace/ficus
python volume2print_sample.py --input_folder workspace/fox/volume/ngp_613 --output_folder workspace/fox/print_volume_s1225/ngp_613

python main_nerf.py data/nerf_synthetic/ficus --workspace workspace/ficus --save_volume --volume_bbox -0.33 -0.36 -0.27 0.33 0.46 0.27 --printing_height 20 --edit_axis 0 2 1
python volume2print_sample.py --input_folder workspace/ficus/volume/ngp_300 --output_folder workspace/ficus/print_volume_s1225/ngp_300
