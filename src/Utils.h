#pragma once
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <hdf5.h>


namespace fs = std::filesystem;

// Note: here we only distinguish UNIX and others (supposing it's Windows)
#ifdef __unix__                   
#define OS_WINDOWS 0
#else     
#define OS_WINDOWS 1
#endif

struct Vertex
{
	// position stored as 4 floats (4th component is supposed to be 1.0)
	Vector4f position;
	// color stored as 4 unsigned char
	Vector4uc color;
};

struct Mesh
{
	// vertices of the mesh
	MatrixX3f vertices;
	// color of the vertices
	MatrixX3f colors;
	// triangulation
	MatrixX3i faces;
};

// replaces the given path with / -> \\ for Windows and viceversa for Unix
static std::string convert_path(std::string path) {
	if (!OS_WINDOWS) std::replace(path.begin(), path.end(), '\\', '/');
	else std::replace(path.begin(), path.end(), '/', '\\');
	return path;
}

// returns the full path to the project root directory
static std::string get_full_path_to_project_root_dir() {
	return { fs::current_path().parent_path().parent_path().u8string() };
}

// tells if the current operating system is windows
static bool isWindows() {
	return OS_WINDOWS == 1;
}

// get shape of the h5 dataset, we assume it has at most dim=2
static std::vector<unsigned int> get_h5_dataset_shape(hid_t h5d) {
	std::vector<unsigned int> shape(2, 0);
	hid_t dspace_id = H5Dget_space(h5d);
	hsize_t dims[2];
	H5Sget_simple_extent_dims(dspace_id, dims, NULL);
	shape[0] = dims[0];
	shape[1] = dims[1];
	return shape;
}