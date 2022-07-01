#pragma once
#include <iostream>
#include <algorithm>
#include <filesystem>

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
	std::vector<Vertex> vertices;
	// triangulation
	std::vector<Vector3i> triangles;
};

// replaces the given path with / -> \\ for Windows and viceversa for Unix
std::string convert_path(std::string path) {
	if (!OS_WINDOWS) std::replace(path.begin(), path.end(), '\\', '/');
	else std::replace(path.begin(), path.end(), '/', '\\');
	return path;
}

// returns the full path to the project root directory
std::string get_full_path_to_project_root_dir() {
	return { fs::current_path().parent_path().parent_path().u8string() };
}

// tells if the current operating system is windows
bool isWindows() {
	return OS_WINDOWS == 1;
}
