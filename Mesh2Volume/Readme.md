/home/arno/Projects/Pint3D/print_ngp/Mesh2Volume

conda activate print3DNerfCUDA124

obj models
1.  /home/arno/Projects/Pint3D/print_ngp/Mesh2Volume/ExampleMesh/colorWheel/color.obj
2. /home/arno/Projects/Pint3D/print_ngp/Mesh2Volume/ExampleMesh/Minecraft_Grass_Block_OBJ/Grass_Block.obj 

渲染贴图到mesh上
python readModel.py --obj_path /home/arno/Projects/Pint3D/print_ngp/Mesh2Volume/ExampleMesh/colorWheel/color.obj

将vertices映射到texture上
python visualize_mesh.py  --obj_path /home/arno/Projects/Pint3D/print_ngp/Mesh2Volume/ExampleMesh/colorWheel/color.obj



Mesh转Volume

MC草方块
python mesh2volme.py --batch-size 5000 --obj_path /home/arno/Projects/Pint3D/print_ngp/Mesh2Volume/ExampleMesh/Minecraft_Grass_Block_OBJ/Grass_Block.obj 
色轮
python mesh2volme.py --batch-size 5000 --obj_path /home/arno/Projects/Pint3D/print_ngp/Mesh2Volume/ExampleMesh/colorWheel/color.obj
