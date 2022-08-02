#pragma once
#include <algorithm>
#include <filesystem>
#include <hdf5.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>


namespace fs = std::filesystem;

// Note: here we only distinguish UNIX and others (supposing it's Windows)
#ifdef __unix__                   
#define OS_WINDOWS 0
#else     
#define OS_WINDOWS 1
#endif

struct Vertex
{
	// position stored as 4 doubles (4th component is supposed to be 1.0)
	Vector4d position;
	// color stored as 4 unsigned char
	Vector4uc color;
};

struct Mesh
{
	// vertices of the mesh
	MatrixX3d vertices;
	// color of the vertices
	MatrixX3d colors;
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

static void mergeBackground(const cv::Mat& originalImg, cv::Mat& renderedImg) {
	int heigh = renderedImg.rows;
	int width = renderedImg.cols;
	for (int i = 0; i < heigh; i++) {
		for (int j = 0; j < width; j++) {
			cv::Vec3b bgr_orig = renderedImg.at<cv::Vec3b>(i, j);
			if (bgr_orig[0] == 0 && bgr_orig[1] == 0 && bgr_orig[2] == 0) {
				cv::Vec3b bgr = originalImg.at<cv::Vec3b>(i, j);
				renderedImg.at<cv::Vec3b>(i, j) = bgr;
			}
		}
	}
}

// add a bounding box with almost black color behind the face (useful for the background merging step to skip the mouse region)
static void addBoundingSquareBehindMouse(VectorXf&vertices, VectorXf&colors, MatrixX3i& triangulation, const VectorXi &landmarks) {
	float color = 0.1;
	float zDisplacement = 15;
	int numVertices = vertices.rows();
	int numTriangulation = triangulation.rows();
	double maxX, maxY, maxZ = -1;
	double minX, minY, minZ = 1e7;
	// get bounding box by iterating over outer lip landmarks
	for (int i = 48; i < 60; i++) {
		double currX = vertices(3*landmarks(i));
		double currY = vertices(3 * landmarks(i) + 1);
		double currZ = vertices(3 * landmarks(i) + 2);
		if (currX > maxX) maxX = currX;
		if (currY > maxY) maxY = currY;
		if (currZ > maxZ) maxZ = currZ;
		if (currX < minX) minX = currX;
		if (currY < minY) minY = currY;
		if (currZ < minZ) minZ = currZ;
	}
	// resize
	vertices.conservativeResize(numVertices + 12, NoChange);
	colors.conservativeResize(numVertices + 12, NoChange);
	triangulation.conservativeResize(numTriangulation + 2, NoChange);
	// v1 top-left
	vertices(numVertices) = minX;
	vertices(numVertices+1) = maxY;
	vertices(numVertices+2) = minZ - zDisplacement;
	// v2 top-right
	vertices(numVertices+3) = maxX;
	vertices(numVertices+4) = maxY;
	vertices(numVertices+5) = minZ - zDisplacement;
	// v3 bottom-left
	vertices(numVertices+6) = minX;
	vertices(numVertices+7) = minY;
	vertices(numVertices+8) = minZ - zDisplacement;
	// v3 bottom-right
	vertices(numVertices+9) = maxX;
	vertices(numVertices+10) = minY;
	vertices(numVertices+11) = minZ - zDisplacement;
	// c1 top-left
	colors(numVertices) = color;
	colors(numVertices + 1) = color;
	colors(numVertices + 2) = color;
	// c2 top-right
	colors(numVertices + 3) = color;
	colors(numVertices + 4) = color;
	colors(numVertices + 5) = color;
	// c3 bottom-left
	colors(numVertices + 6) = color;
	colors(numVertices + 7) = color;
	colors(numVertices + 8) = color;
	// c4 bottom-right
	colors(numVertices + 9) = color;
	colors(numVertices + 10) = color;
	colors(numVertices + 11) = color;
	// add triangulation
	// t1
	triangulation(numTriangulation, 0) = int(numVertices / 3);
	triangulation(numTriangulation, 1) = int(numVertices / 3)+2;
	triangulation(numTriangulation, 2) = int(numVertices / 3)+1;
	// t2
	triangulation(numTriangulation+1, 0) = int(numVertices / 3)+1;
	triangulation(numTriangulation+1, 1) = int(numVertices / 3)+2;
	triangulation(numTriangulation+1, 2) = int(numVertices / 3)+3;
}