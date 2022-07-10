#pragma once
#include "DataHandler.h"

// Class to store information of a RGBD Image and the detected landmarks.
class Image {
public:
	Image() {
	}
	// Constructor that loads all necessary information from the files via DataHandler. If _fileName is not specified, "sample1" is taken as default.
	Image(std::string _fileName) {
		fileName = _fileName;
		DataHandler::loadDepthMap(_fileName, depthMap);
		rgb = std::vector<MatrixXd>(3);
		DataHandler::loadRGB(_fileName, rgb);
		DataHandler::loadLandmarks(fileName, landmarks);
		width = depthMap.rows();
		height = depthMap.cols();
	}

	// getters

	std::string getFileName() { return fileName; }

	unsigned int getWidth() { return width; }

	unsigned int getHeight() { return height; }

	MatrixXd getDepthMap() { return depthMap; }

	MatrixXd getLandmarks() { return landmarks; }

	Vector2d getLandmark(unsigned i) { return landmarks.row(i); }

	std::vector<MatrixXd> detRGB() { return rgb; }

private:
	std::string fileName;	// name of the image file
	unsigned int width, height;	// size of the image
	MatrixXd depthMap;	// matrix of W x H. 255 for closest distance and 0 for farthest.
	std::vector<MatrixXd> rgb;	// [r, g, b] each of these is a matrix of W x H with values between [0, 255]
	MatrixX2d landmarks;	// 68 2D facial landmarks detected by dl model
};