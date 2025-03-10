
"""
Ten写的基于VTK的3D体积查看器
"""
import vtk
import numpy as np
from PIL import Image
import os
import argparse

class VolumeViewer:
    def __init__(self, image_dir, x_scale=1.0, y_scale=1.0, z_scale=1.0, use_rgba=False, transparent=False, render=False):
        # Load all images
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        if not image_files:
            raise ValueError("No PNG images found in the directory")
            
        # Load first image to get dimensions
        first_image = Image.open(os.path.join(image_dir, image_files[0]))
        width, height = first_image.size
        depth = len(image_files)
        
        if use_rgba:
            # Create 4D numpy array for RGBA
            volume_array = np.zeros((depth, height, width, 4), dtype=np.uint8)
            
            # Load all images into the array
            for i, img_file in enumerate(image_files):
                img = Image.open(os.path.join(image_dir, img_file)).convert('RGBA')
                volume_array[i] = np.array(img)
                
            # Create VTK data objects
            self.dataImporter = vtk.vtkImageImport()
            data_string = volume_array.tobytes()
            self.dataImporter.CopyImportVoidPointer(data_string, len(data_string))
            self.dataImporter.SetDataScalarTypeToUnsignedChar()
            self.dataImporter.SetNumberOfScalarComponents(4)  # RGBA = 4 components
            
        else:
            # Original grayscale mode
            volume_array = np.zeros((depth, height, width), dtype=np.uint8)
            
            # Load all images into the array
            for i, img_file in enumerate(image_files):
                img = Image.open(os.path.join(image_dir, img_file)).convert('RGBA')
                img_array = np.array(img)
                
                # Create mask for red pixels (255, 0, 0, 255)
                red_mask = np.all(img_array == [255, 0, 0, 255], axis=-1)
                
                # Create mask for transparent pixels (0, 0, 0, 0)
                transparent_mask = np.all(img_array == [0, 0, 0, 0], axis=-1)
                
                # Modify the img_array based on masks
                img_array[red_mask] = [0, 0, 0, 0]  # Set red pixels to transparent
                if not transparent:
                    img_array[transparent_mask] = [0, 0, 0, 25]  # Set transparent pixels to slight opacity
                
                # Convert to grayscale array (using alpha channel)
                gray_array = img_array[..., 3]  # Take the alpha channel as grayscale
                volume_array[i] = gray_array
                
            # Create VTK data objects
            self.dataImporter = vtk.vtkImageImport()
            data_string = volume_array.tobytes()
            self.dataImporter.CopyImportVoidPointer(data_string, len(data_string))
            self.dataImporter.SetDataScalarTypeToUnsignedChar()
            self.dataImporter.SetNumberOfScalarComponents(1)  # Grayscale = 1 component
        
        self.dataImporter.SetDataExtent(0, width-1, 0, height-1, 0, depth-1)
        self.dataImporter.SetWholeExtent(0, width-1, 0, height-1, 0, depth-1)
        self.dataImporter.SetDataSpacing(x_scale, y_scale, z_scale)
        
        # Create volume property
        self.volumeProperty = vtk.vtkVolumeProperty()
        self.volumeProperty.ShadeOn()  # Enable shading
        
        # # Adjust lighting parameters for better color preservation
        # self.volumeProperty.SetAmbient(0.4)    # Reduce ambient to avoid oversaturation
        # self.volumeProperty.SetDiffuse(0.6)    # Add some diffuse for depth
        # self.volumeProperty.SetSpecular(0.2)   # Minimal specular to avoid bright spots
        # self.volumeProperty.SetSpecularPower(10.0)  # Control the size of specular highlights
        
        if use_rgba:
            # Set up RGBA transfer functions
            self.volumeProperty.IndependentComponentsOn()
            
            # Set up transfer functions for each component (R,G,B,A)
            for i in range(4):
                colorFunc = vtk.vtkColorTransferFunction()
                opacityFunc = vtk.vtkPiecewiseFunction()
                
                if i == 0:  # Red component
                    colorFunc.AddRGBPoint(0, 0.0, 0.0, 0.0)
                    colorFunc.AddRGBPoint(255, 1.0, 0.0, 0.0)  # Full red
                elif i == 1:  # Green component
                    colorFunc.AddRGBPoint(0, 0.0, 0.0, 0.0)
                    colorFunc.AddRGBPoint(255, 0.0, 1.0, 0.0)  # Full green
                elif i == 2:  # Blue component
                    colorFunc.AddRGBPoint(0, 0.0, 0.0, 0.0)
                    colorFunc.AddRGBPoint(255, 0.0, 0.0, 1.0)  # Full blue
                else:  # Alpha component
                    colorFunc.AddRGBPoint(0, 1.0, 1.0, 1.0)  # White for alpha
                    colorFunc.AddRGBPoint(255, 1.0, 1.0, 1.0)
                
                # Set opacity for all components
                opacityFunc.AddPoint(0, 0.0)
                if i == 3:  # Alpha component
                    if transparent:
                        opacityFunc.AddPoint(0, 0.0)  # Fully transparent
                    else:
                        opacityFunc.AddPoint(25, 0.1)  # Low alpha values slightly visible
                opacityFunc.AddPoint(255, 1.0)
                
                self.volumeProperty.SetColor(i, colorFunc)
                self.volumeProperty.SetScalarOpacity(i, opacityFunc)
        else:
            # Original grayscale transfer functions
            colorFunc = vtk.vtkColorTransferFunction()
            colorFunc.AddRGBPoint(0, 0.0, 0.0, 0.0)
            colorFunc.AddRGBPoint(128, 0.5, 0.5, 0.5)
            colorFunc.AddRGBPoint(255, 1.0, 1.0, 1.0)
            self.volumeProperty.SetColor(colorFunc)
            
            opacityFunc = vtk.vtkPiecewiseFunction()
            opacityFunc.AddPoint(0, 0.0)
            opacityFunc.AddPoint(128, 0.5)
            opacityFunc.AddPoint(255, 1.0)
            self.volumeProperty.SetScalarOpacity(opacityFunc)

        # Create volume mapper
        self.volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
        self.volumeMapper.SetInputConnection(self.dataImporter.GetOutputPort())
        
        # Create volume
        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(self.volumeMapper)
        self.volume.SetProperty(self.volumeProperty)
        
        # Create renderer and render window
        self.renderer = vtk.vtkRenderer()
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.SetMultiSamples(8)  # Enable anti-aliasing with 8x MSAA
        self.renderWindow.AddRenderer(self.renderer)
        
        # Add volume to renderer
        self.renderer.AddVolume(self.volume)
        
        # Create interactor
        self.renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        self.renderWindowInteractor.SetRenderWindow(self.renderWindow)
        
        # Set background color and size
        self.renderer.SetBackground(0.1, 0.1, 0.1)
        if render:
            # self.renderWindow.SetSize(1920, 1080)  # 1080p resolution
            self.renderWindow.SetSize(600, 400)  # 1080p resolution
            self.renderWindow.SetAlphaBitPlanes(1)  # Enable alpha channel
            self.renderWindow.SetMultiSamples(8)    # Anti-aliasing
            self.renderer.SetBackground(0.0, 0.0, 0.0)  # Black background
            self.renderer.GradientBackgroundOff()   # Disable gradient background
        else:
            self.renderWindow.SetSize(800, 800)
        
        # Initialize camera
        self.renderer.ResetCamera()
        
    def render_image(self, output_path="render.png"):
        # Create window to image filter
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(self.renderWindow)
        w2if.SetInputBufferTypeToRGBA()    # Capture alpha channel
        w2if.Update()

        # Write to file
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(output_path)
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.Write()

    def start(self, render=False):
        self.renderWindowInteractor.Initialize()
        self.renderWindow.Render()
        
        if render:
            self.render_image()
        else:
            self.renderWindowInteractor.Start()

def main():
    parser = argparse.ArgumentParser(description='View 3D volume from stack of PNG images')
    parser.add_argument('--image_dir', type=str, default=r"M:\Projects\3DPrinting\shape_padding\padded", help='Directory containing PNG image files')
    parser.add_argument('--x-scale', type=float, default=0.042333, help='Scale factor for width (default: 1.0)')
    parser.add_argument('--y-scale', type=float, default=0.0846666, help='Scale factor for height (default: 1.0)')
    parser.add_argument('--z-scale', type=float, default=0.014, help='Scale factor for layer spacing (default: 1.0)')
    parser.add_argument('--rgba', action='store_true', help='Use RGBA mode instead of grayscale')
    parser.add_argument('--transparent', action='store_true', help='Allow fully transparent pixels (alpha=0) to be invisible')
    parser.add_argument('--render', action='store_true', help='Render a high-resolution image with transparent background')
    
    args = parser.parse_args()
    
    # Generate output filename based on input directory
    folder_name = os.path.basename(os.path.normpath(args.image_dir))
    output_path = f"./{folder_name}_volume_output.png"
    
    viewer = VolumeViewer(
        args.image_dir,
        x_scale=args.x_scale,
        y_scale=args.y_scale,
        z_scale=args.z_scale,
        use_rgba=args.rgba,
        transparent=args.transparent,
        render=args.render
    )
    viewer.start(render=args.render)

if __name__ == "__main__":
    main() 