python main_nerf.py data/nerf_synthetic/lego --workspace workspace/lego --save_volume --volume_bbox -0.39 -0.13 -0.26 0.39 0.39 0.27 --printing_height 20 --edit_axis 0 2 1
python main_nerf.py data/nerf_synthetic/lego --workspace workspace/lego --test --gui

python main_nerf.py data/fox --workspace workspace/fox --save_volume --volume_bbox  -0.74 -0.76 -0.22 0.34 0.64 0.78 --printing_height 20 --edit_axis 0 1 2
python main_nerf.py data/fox --workspace workspace/fox --test --gui
