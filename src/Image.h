#pragma once
#include "DataHandler.h"

// Class to store information of a RGBD Image and the detected landmarks.
class Image {
public:
	Image() {
	}

	// Constructor that loads all necessary information from the files via DataHandler. If _fileName is not specified, "sample1" is taken as default.
	Image(std::string _fileName, double _downscale = 2) {
		downscale = _downscale;
		fileName = _fileName;
		DataHandler::loadDepthMap(_fileName, depthMap, depthMapDown);
		rgb = std::vector<MatrixXd>(3);
		rgbDown = std::vector<MatrixXd>(3);
		DataHandler::loadRGB(_fileName, rgb, rgbDown);
		DataHandler::loadLandmarks(fileName, landmarks);
		height = depthMap.rows();
		width = depthMap.cols();
		normalMap = cv::Mat(height, width, CV_64FC3);
		normalMapDown = cv::Mat(int(height / downscale), int(width / downscale), CV_64FC3);
		computeNormals();
	}

	// getters

	std::string getFileName() { return fileName; }

	unsigned int getWidth() { return width; }

	unsigned int getHeight() { return height; }

	unsigned int getWidthDown() { return unsigned int(width / downscale); }

	unsigned int getHeightDown() { return unsigned int(height / downscale); }

	MatrixXd getDepthMap() { return depthMap; }

	MatrixXd getDepthMapDown() { return depthMapDown; }

	MatrixXd getLandmarks() { return landmarks; }

	Vector2d getLandmark(unsigned i) { return landmarks.row(i); }

	cv::Mat getNormalMap() { return normalMap; }

	cv::Mat getNormalMapDown() { return normalMapDown; }

	std::vector<MatrixXd> getRGB() { return rgb; }
	
	std::vector<MatrixXd> getRGBDown() { return rgbDown; }

	double getDownScale() { return downscale; }

private:
	void computeNormals() {
		int height = depthMap.rows();
		int width = depthMap.cols();

		for (int v = 1; v < height - 1; ++v) {
			for (int u = 1; u < width - 1; ++u) {
				Vector3d v_top = Vector3d(u, v - 1, double(depthMap(v - 1, u)) / 255.);
				Vector3d v_down = Vector3d(u, v + 1, double(depthMap(v + 1, u)) / 255.);
				Vector3d u_left = Vector3d(u - 1, v, double(depthMap(v, u - 1)) / 255.);
				Vector3d u_right = Vector3d(u + 1, v, double(depthMap(v, u + 1)) / 255.);
				Vector3d normal = -(u_right - u_left).cross(v_down - v_top);
				normal.normalize();
				normalMap.at<cv::Vec3d>(v, u)[0] = normal(0);
				normalMap.at<cv::Vec3d>(v, u)[1] = normal(1);
				normalMap.at<cv::Vec3d>(v, u)[2] = normal(2);
			}
		}

		height = depthMapDown.rows();
		width = depthMapDown.cols();

		for (int v = 1; v < height - 1; ++v) {
			for (int u = 1; u < width - 1; ++u) {
				Vector3d v_top = Vector3d(u, v - 1, double(depthMapDown(v - 1, u)) / 255.);
				Vector3d v_down = Vector3d(u, v + 1, double(depthMapDown(v + 1, u)) / 255.);
				Vector3d u_left = Vector3d(u - 1, v, double(depthMapDown(v, u - 1)) / 255.);
				Vector3d u_right = Vector3d(u + 1, v, double(depthMapDown(v, u + 1)) / 255.);
				Vector3d normal = -(u_right - u_left).cross(v_down - v_top);
				normal.normalize();
				normalMapDown.at<cv::Vec3d>(v, u)[0] = normal(0);
				normalMapDown.at<cv::Vec3d>(v, u)[1] = normal(1);
				normalMapDown.at<cv::Vec3d>(v, u)[2] = normal(2);
			}
		}
	}

	std::string fileName;	// name of the image file
	unsigned int width, height;	// size of the image
	MatrixXd depthMap, depthMapDown;	// matrix of HxW. 255 for closest distance and 0 for farthest.
	std::vector<MatrixXd> rgb, rgbDown;	// [r, g, b] each of these is a matrix of W x H with values between [0, 255]
	MatrixX2d landmarks;	// 68 2D facial landmarks detected by dl model
	cv::Mat normalMap, normalMapDown;
	double downscale;
};