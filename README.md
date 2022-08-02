# FaceReconstruction
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
2. Install OpenGL libraries **glfw3** and **glew**.
```
sudo apt-get install libglew-dev
sudo apt-get install libglfw3
sudo apt-get install libglfw3-dev
```
3. Set your current working dictory to the root folder of this project, run the following code and do not close the terminal until the installation
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
Some files needs to be placed in certain directories so the program can find them correctly:
```
- data/
    BFM17.h5      # Basel Face Model 2017, which can be downloaded from "https://faces.dmi.unibas.ch/bfm/bfm2017.html" (the simplified model)
```
### Preprocessin scripts

#### Landmarks
You have to run /scripts/extractLandmarks.py to precompute the location of the landmarks for the input image (or sequence), before trying to run the program and fit the face model.

#### Depth
Depth map of the input should be provided in data/depth/ folder. Otherwise you can use /scripts/predictDepth.py to estimate the depth using a DL model.

#### Sequence preprocessing
We also provide a script preprocessSequence.py that we used to center crop and modify the depth map captured with a kinect to fit our case.
