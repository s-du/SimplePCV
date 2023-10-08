  # SimplePCV
Creating Ambient occlusion render of point clouds (elevation maps, DSM or DTM) using "Portion de Ciel Visible" (PCV) algorithm.
A small PySide6 GUI allows to play with the colormap range (using matplotlib 'Greys' by default).

## Introduction
Making convincing ambient occlusion renders from elevation maps (terrains or construction sites, for example) can be useful for better point cloud segmentation. Using CloudCompare can be overkill for such a task.
Based on Duguet, Florent & Girardeau-Montaut, Daniel. (2004). Rendu en Portion de Ciel Visible de Gros Nuages de Points 3D.

NOTE: creating the input dtm tif can be done with CloudCompare or any point cloud rasterization tools.

\#Pillow \#Open3D \#Voxels \#ImageProcessing \#Pointcloud 

## Use
Install all requirements, then simply run main.py (Do not forget to change the image path).
One test image is provided

<p align="center">
    <a href="https://ibb.co/G3YVhdP"><img src="https://i.ibb.co/S7YKL6v/montage-pcv.png" alt="montage-pcv" border="0"></a>
  
    Simple PCV (ambient occlusion)
</p>


## Installation instructions

1. Clone the repository:
```
git clone https://github.com/s-du/SimplePCV
```

2. Navigate to the app directory:
```
cd SimplePCV
```

3. Install the required dependencies:
```
pip install -r requirements.txt
```
4. (optional) Modify main.py --> replace image path
   
6. Run the app:
```
python main.py
```

## Contributing

Contributions to the app are welcome! If you find any bugs, have suggestions for new features, or would like to contribute enhancements, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make the necessary changes and commit them.
4. Push your changes to your fork.
5. Submit a pull request describing your changes.
