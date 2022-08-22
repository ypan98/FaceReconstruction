# FaceReconstruction

This is work from the couse 3D Scanning and Motion Capture @TUM [IN2354]. The goal of the project is to do facial expression transfer from one actor to another. For this end, a 3D parametric face model is fit (optimized) w.r.t. the input RGB-D sequences in a non-linear least squares way, following the approach of Justus et al. 2015.

To see the results, check the images in papers/final_report.

<strong>Authors</strong>
1. Yimin Pan
2. Weixiao Xia
3. Wei Yi
4. Xiyue Zhang

## Install Dependencies
In order to install dependencies for this project please follow the following steps.
### For Ubuntu Users
1. Run the following code in your terminal in order to install the installation tools
```
sudo apt-get update
sudo apt-get install cmake gcc g++ dos2unix

sudo apt install -y ccache
sudo /usr/sbin/update-ccache-symlinks
echo 'export PATH="/usr/lib/ccache:$PATH"' | tee -a ~/.bashrc
source ~/.bashrc && echo $PATH
```
2. Set your current working dictory to the root folder of this project, run the following code and do not close the terminal until the installation
is done.
```
dos2unix install_dependencies_linux.sh
sudo bash install_dependencies_linux.sh
```
### For Windows Users
#### If you'd like to run this project with Visual Studio
1. Make sure you have Git Bash installed in your computer.
2. Open Git Bash
3. Set your current working dictory to the root folder of this project
4. Find your Visual Studio's version. For example, "Visual Studio 16 2019"
5. Run the following code with the right Visual Studio version found in the previous step
```
./install_dependencies_win.sh "Visual Studio 16 2019"
```
#### If you'd like to run this project under WSL
1. Install the Ubutu 20.04.4 LTS wsl from Microsoft Store. (You will have to reboot your system in order to finish the installation process)
2. Open Windows PowerShell and activate the WSL using the following command
```
wsl
```
3. Once you have activated wsl switch the current working directory to the root folder of this project and follow the same process described 
for ubuntu users. (hint: to switch to disk C for example you can use cd /mnt/c/)

### For Both Users
Our method uses a cuda-parallelized rasterizer so in order to run this project you need to have a nvidia GPU with CUDA installed in your system.

## Build the project
### For Visual Studio users
You have to copy all the .dll files of libraries glog, opencv and hdf5 to your executable's folder. After this you will be able build the target **face_reconstruction**.
### For WSL and Linux-system users
1. Go to the **src** folder
2. Create **build** folder with command
```
mkdir build
```
3. Go to the created **build** folder
```
cd build
```
4. Compile and build the project (you can also build this project in debug mode, just need to replace **Release** by **Debug**)
```
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

## Data Preparation
Some files needs to be placed in certain directories so the program can find them correctly.
###  The face model dataset
Containing informations like basis, mean, std...
```
- data/
    BFM17.h5      # Basel Face Model 2017, which can be downloaded from "https://faces.dmi.unibas.ch/bfm/bfm2017.html" (the simplified model)
```
### Input image (RGB-D)
Basically the RGB and depth map image from which the face mesh is reconstructed.
```
- data/
    samples/
        depth/
            sample.png      # Sample depth map
        rgb/
            sample.png      # Sample RGB 
            
Landmarks are needed too, but this can be predicted from the RGB input with the provided script.
```

### Preprocessin scripts
The scripts are implemented in Python and you would need to install some packages to execute them:
```
cd scripts
pip install -r requirements.txt
```
#### Landmarks
You have to run /scripts/extractLandmarks.py to precompute the location of the landmarks for the input image (or sequence), before trying to run the program and fit the face model.

#### Depth
If you dont have the depth map but only the RGB image, an option is to use /scripts/predictDepth.py to estimate it using a DL model. Although the result is much noisier compared to one that is captured by depth sensors. Therefore, the reconstruction result is no that great.

#### Sequence preprocessing
We also provide a script preprocessSequence.py that we used to center crop and adapt the depth map captured with a kinect to fit our case. 
