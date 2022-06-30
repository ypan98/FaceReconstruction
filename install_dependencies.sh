#!/usr/bin/env bash
PWD=$(pwd)

# Activate debug settings
set -x
set -e

GCC=gcc
GXX=g++

# The default build type is set to RelWithDebInfo, but we can change it later by
# providing an extra argument
BUILD_TYPE=RelWithDebInfo

if [ -n "$1" ]; then
BUILD_TYPE=$1
fi

# Get the number of cores which will be later used for configuring the compiling setting
NUM_CORES=`getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu || echo 1`

NUM_PARALLEL_BUILDS=$((NUM_CORES - 2 < 1 ? 1 : NUM_CORES - 2))

# Set the default cmake params
CXX_MARCH=native

COMMON_CMAKE_ARGS=(
    -DCMAKE_C_COMPILER=${GCC}
    -DCMAKE_CXX_COMPILER=${GXX}
    -DCMAKE_C_COMPILER_LAUNCHER=ccache
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
    -DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON
    -DCMAKE_CXX_FLAGS="-march=$CXX_MARCH -O3 -Wno-deprecated-declarations -Wno-null-pointer-arithmetic -Wno-unknown-warning-option -Wno-unused-function"
)

# Explicitly synchronize the submodules in case they are not pulled to local yet.
git submodule update --init --recursive

########### Installing the eigen library ###########
EIGEN_LIB="$PWD/libs/eigen"
EIGEN_BUILD="$PWD/eigen/build"

if [ ! -d "$EIGEN_LIB" ]; then
mkdir -p "$EIGEN_BUILD"
mkdir -p "$EIGEN_LIB"

pushd "$EIGEN_BUILD"
cmake .. "${COMMON_CMAKE_ARGS[@]}" \
-DCMAKE_INSTALL_PREFIX="$EIGEN_LIB" \
-DBUILD_TESTING=OFF
make -j$NUM_PARALLEL_BUILDS install
popd
fi

########### Installing the glog library ###########
GLOG_LIB="$PWD/libs/glog"
GLOG_BUILD="$PWD/glog/build"

if [ ! -d "$GLOG_LIB" ]; then
mkdir -p "$GLOG_LIB"
mkdir -p "$GLOG_BUILD"

pushd "$GLOG_BUILD"
cmake .. "${COMMON_CMAKE_ARGS[@]}" \
-DCMAKE_INSTALL_PREFIX="$GLOG_LIB" \
-DBUILD_TESTING=OFF \
-DWITH_GFLAGS=OFF
make -j$NUM_PARALLEL_BUILDS install
popd
fi

########### Installing the ceres library ###########
# Important: glog is a dependency of ceres
CERES_LIB="$PWD/libs/ceres"
CERES_BUILD="$PWD/ceres-solver/build_dir"

if [ ! -d "$CERES_LIB" ]; then
mkdir -p "$CERES_LIB"
mkdir -p "$CERES_BUILD"

pushd "$CERES_BUILD"
cmake .. "${COMMON_CMAKE_ARGS[@]}" \
-DCMAKE_INSTALL_PREFIX="$CERES_LIB" \
-DBUILD_TESTING=OFF \
-DBUILD_EXAMPLES=OFF \
-Dglog_DIR="$GLOG_LIB/lib/cmake/glog" \
-DEigen3_DIR="$EIGEN_LIB/share/eigen3/cmake"
make -j$NUM_PARALLEL_BUILDS install
popd
fi

########### Installing the opencv library ###########
OPENCV_LIB="$PWD/libs/opencv"
OPENCV_BUILD="$PWD/opencv/build"

if [ ! -d "$OPENCV_LIB" ]; then
mkdir -p "$OPENCV_LIB"
mkdir -p "$OPENCV_BUILD"

pushd "$OPENCV_BUILD"
cmake .. "${COMMON_CMAKE_ARGS[@]}" \
-DCMAKE_INSTALL_PREFIX="$OPENCV_LIB" \
-DBUILD_TESTS=OFF \
-DBUILD_PERF_TESTS=OFF \
-DBUILD_EXAMPLES=OFF \
-DBUILD_opencv_apps=OFF
make -j$NUM_PARALLEL_BUILDS install
popd
fi

########### Installing the hdf5 library ###########
HDF5_LIB="$PWD/libs/hdf5"
HDF5_BUILD="$PWD/hdf5/build"

if [ ! -d "$HDF5_LIB" ]; then
mkdir -p "$HDF5_LIB"
mkdir -p "$HDF5_BUILD"

pushd "$HDF5_BUILD"
cmake .. "${COMMON_CMAKE_ARGS[@]}" \
-DCMAKE_INSTALL_PREFIX="$HDF5_LIB" \
-DBUILD_TESTING=OFF \
-DBUILD_EXAMPLES=OFF
make -j$NUM_PARALLEL_BUILDS install
popd
fi

############ Delete the source and the build files of the dependencies ##############
EIGEN_SOURCE="$PWD/libs/eigen"
GLOG_SOURCE="$PWD/libs/glog"
CERES_SOURCE="$PWD/libs/ceres-solver"
OPENCV_SOURCE="$PWD/libs/opencv"
HDF5_SOURCE="$PWD/libs/hdf5"


