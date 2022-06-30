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
2. Set your current working dictory to the root folder of this project, run the following code and do not close the terminal until the installation
is done.
```
dos2unix install_dependencies.sh
sudo bash install_dependencies.sh
```

### For Windows Users
1. Install the Ubutu 20.04.4 LTS wsl from Microsoft Store. (You will have to reboot your system in order to finish the installation process)
2. Open Windows PowerShell and activate the WSL using the following command

```
wsl
```
3. Once you have activated wsl switch the current working directory to the root folder of this project and follow the same process described 
for ubuntu users. (hint: to switch to disk C for example you can use cd /mnt/c/)
