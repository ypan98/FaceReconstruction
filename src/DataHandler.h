#pragma once

#include <iostream>
#include <fstream>
#include "Eigen.h"

// Note: here we only distinguish UNIX and others (supposing it's Windows)
#ifdef __unix__                   
	#define OS_WINDOWS 0
#else     
	#define OS_WINDOWS 1
#endif


std::string basePath = "D:\\TUM\\FaceReconstruction\\samples\\landmark\\";
unsigned int NUM_LANDMARKS = 68; // num of landmark points
unsigned int LANDMARK = 2; // each landmark is a 2D point


class DataHandler{
public:
	// read the precomputed landmarks from the file 
	static MatrixXf get_landmarks(string fileName) {

	}
	// get [width, depth] of the image
	static Vector2i get_image_size(string fileName) {

	}
	// read rgb value of the pixels from the image
	static MatrixXf get_rgb(string fileName) {

	}
	// read the depth map of the image
	static MatrixXf get_depth(string fileName) {

	}


}