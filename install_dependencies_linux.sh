#!/usr/bin/env bash
PWD=$(pwd)

# Activate debug settings
set -x
set -e

# Get the number of cores which will be later used for configuring the compiling setting
NUM_CORES=`getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu || echo 1`

NUM_PARALLEL_BUILDS=$((NUM_CORES - 2 < 1 ? 1 : NUM_CORES - 2))

# Set the default cmake params
CXX_MARCH=native

# Explicitly synchronize the submodules in case they are not pulled to local yet.
git submodule update --init --recursive

########### Installing the eigen library ###########
EIGEN_LIB="$PWD/libs/eigen"
EIGEN_BUILD="$PWD/dependencies/eigen/build"

if [ ! -d "$EIGEN_LIB" ]; then
mkdir -p "$EIGEN_BUILD"
mkdir -p "$EIGEN_LIB"

pushd "$EIGEN_BUILD"
cmake .. "${CMAKE_GENERATOR_OPTIONS[@]}" \
-DCMAKE_INSTALL_PREFIX="$EIGEN_LIB" \
-DBUILD_TESTING=OFF
cmake --build . -j$NUM_PARALLEL_BUILDS --target install
popd
fi

########### Installing the glog library ###########
GLOG_LIB="$PWD/libs/glog"

GLOG_BUILD_DEBUG="$PWD/dependencies/glog/build_debug"
GLOG_BUILD_RELEASE="$PWD/dependencies/glog/build_release"

if [ ! -d "$GLOG_LIB" ]; then
mkdir -p "$GLOG_LIB"
mkdir -p "$GLOG_BUILD_DEBUG"
mkdir -p "$GLOG_BUILD_RELEASE"


pushd "$GLOG_BUILD_DEBUG"
cmake .. -DCMAKE_INSTALL_PREFIX="$GLOG_LIB" \
-DBUILD_TESTING=OFF \
-DWITH_GFLAGS=OFF \
-DCMAKE_BUILD_TYPE=DEBUG
make -j$NUM_PARALLEL_BUILDS install
popd

pushd "$GLOG_BUILD_RELEASE"
cmake .. -DCMAKE_INSTALL_PREFIX="$GLOG_LIB" \
-DBUILD_TESTING=OFF \
-DWITH_GFLAGS=OFF \
-DCMAKE_BUILD_TYPE=RELEASE
make -j$NUM_PARALLEL_BUILDS install
popd
fi

########### Installing the ceres library ###########
# Important: glog is a dependency of ceres
CERES_LIB="$PWD/libs/ceres"

CERES_BUILD_DEBUG="$PWD/dependencies/ceres-solver/build_debug"
CERES_BUILD_RELEASE="$PWD/dependencies/ceres-solver/build_release"

if [ ! -d "$CERES_LIB" ]; then
mkdir -p "$CERES_LIB"
mkdir -p "$CERES_BUILD_DEBUG"
mkdir -p "$CERES_BUILD_RELEASE"

pushd "$CERES_BUILD_DEBUG"
cmake .. -DCMAKE_INSTALL_PREFIX="$CERES_LIB" \
-DBUILD_TESTING=OFF \
-DBUILD_EXAMPLES=OFF \
-Dglog_DIR="$GLOG_LIB_DEBUG/lib/cmake/glog" \
-DEigen3_DIR="$EIGEN_LIB/share/eigen3/cmake" \
-DCMAKE_BUILD_TYPE=DEBUG
make -j$NUM_PARALLEL_BUILDS install
popd

pushd "$CERES_BUILD_RELEASE"
cmake .. -DCMAKE_INSTALL_PREFIX="$CERES_LIB" \
-DBUILD_TESTING=OFF \
-DBUILD_EXAMPLES=OFF \
-Dglog_DIR="$GLOG_LIB_RELEASE/lib/cmake/glog" \
-DEigen3_DIR="$EIGEN_LIB/share/eigen3/cmake" \
-DCMAKE_BUILD_TYPE=RELEASE

make -j$NUM_PARALLEL_BUILDS install
popd
fi

########### Installing the opencv library ###########
OPENCV_LIB="$PWD/libs/opencv"
OPENCV_BUILD_DEBUG="$PWD/dependencies/opencv/build_debug"
OPENCV_BUILD_RELEASE="$PWD/dependencies/opencv/build_release"

if [ ! -d "$OPENCV_LIB" ]; then
mkdir -p "$OPENCV_LIB"
mkdir -p "$OPENCV_BUILD_DEBUG"
mkdir -p "$OPENCV_BUILD_RELEASE"

pushd "$OPENCV_BUILD_DEBUG"
cmake .. -DCMAKE_INSTALL_PREFIX="$OPENCV_LIB" \
-DBUILD_TESTS=OFF \
-DBUILD_PERF_TESTS=OFF \
-DBUILD_EXAMPLES=OFF \
-DBUILD_DOCS=OFF \
-DWITH_CUDA=OFF \
-DBUILD_opencv_apps=OFF \
-DBUILD_opencv_gapi=OFF \
-DBUILD_opencv_dnn=OFF \
-DBUILD_opencv_java=OFF \
-DBUILD_opencv_python2=OFF \
-DBUILD_opencv_python3=OFF \
-DCMAKE_BUILD_TYPE=DEBUG
make -j$NUM_PARALLEL_BUILDS install
popd

pushd "$OPENCV_BUILD_RELEASE"
cmake .. -DCMAKE_INSTALL_PREFIX="$OPENCV_LIB" \
-DBUILD_TESTS=OFF \
-DBUILD_PERF_TESTS=OFF \
-DBUILD_EXAMPLES=OFF \
-DBUILD_DOCS=OFF \
-DWITH_CUDA=OFF \
-DBUILD_opencv_apps=OFF \
-DBUILD_opencv_gapi=OFF \
-DBUILD_opencv_dnn=OFF \
-DBUILD_opencv_java=OFF \
-DBUILD_opencv_python2=OFF \
-DBUILD_opencv_python3=OFF \
-DCMAKE_BUILD_TYPE=RELEASE

make -j$NUM_PARALLEL_BUILDS install
popd
fi

########### Installing the hdf5 library ###########
HDF5_LIB="$PWD/libs/hdf5"
HDF5_BUILD_DEBUG="$PWD/dependencies/hdf5/build_debug"
HDF5_BUILD_RELEASE="$PWD/dependencies/hdf5/build_release"

if [ ! -d "$HDF5_LIB" ]; then
mkdir -p "$HDF5_LIB"
mkdir -p "$HDF5_BUILD_DEBUG"
mkdir -p "$HDF5_BUILD_RELEASE"

pushd "$HDF5_BUILD_DEBUG"
cmake .. -DCMAKE_INSTALL_PREFIX="$HDF5_LIB" \
-DBUILD_TESTING=OFF \
-DBUILD_EXAMPLES=OFF
make -j$NUM_PARALLEL_BUILDS install
popd

pushd "$HDF5_BUILD_RELEASE"
cmake .. -DCMAKE_INSTALL_PREFIX="$HDF5_LIB" \
-DBUILD_TESTING=OFF \
-DBUILD_EXAMPLES=OFF
make -j$NUM_PARALLEL_BUILDS install
popd
fi

############ Delete the source and the build files of the dependencies ##############
EIGEN_SOURCE="$PWD/dependencies/eigen"
GLOG_SOURCE="$PWD/dependencies/glog"
CERES_SOURCE="$PWD/dependencies/ceres-solver"
OPENCV_SOURCE="$PWD/dependencies/opencv"
HDF5_SOURCE="$PWD/dependencies/hdf5"

git submodule deinit -f "$EIGEN_SOURCE"
git submodule deinit -f "$GLOG_SOURCE"
git submodule deinit -f "$CERES_SOURCE"
git submodule deinit -f "$OPENCV_SOURCE"
git submodule deinit -f "$HDF5_SOURCE"

# In case the deinit command didn't cleared the submodule's folder

EIGEN_SOURCE="./dependencies/eigen"
GLOG_SOURCE="./dependencies/glog"
CERES_SOURCE="./dependencies/ceres-solver"
OPENCV_SOURCE="./dependencies/opencv"
HDF5_SOURCE="./dependencies/hdf5"

if [ -z "$(ls -A $EIGEN_SOURCE)" ]; then
    echo "Empty"
else
    pushd "$EIGEN_SOURCE"
    rm -r *
    rm -r *
    popd
fi

if [ -z "$(ls -A $GLOG_SOURCE)" ]; then
    echo "Empty"
else
    pushd "$GLOG_SOURCE"
    rm -r *
    rm -r *
    popd
fi

if [ -z "$(ls -A $CERES_SOURCE)" ]; then
    echo "Empty"
else
    pushd "$CERES_SOURCE"
    rm -r *
    rm -r *
    popd
fi

if [ -z "$(ls -A $OPENCV_SOURCE)" ]; then
    echo "Empty"
else
    pushd "$OPENCV_SOURCE"
    rm -r *
    rm -r *
    popd
fi

if [ -z "$(ls -A $HDF5_SOURCE)" ]; then
    echo "Empty"
else
    pushd "$HDF5_SOURCE"
    rm -r *
    rm -r *
    popd
fi
