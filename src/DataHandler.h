#pragma once

#include <fstream>
#include "Eigen.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <vector>
#include "Utils.h"
#include <map>

#define NUM_LANDMARKS 68
#define LANDMARK_DIM 2

// Full paths
const std::string PATH_TO_LANDMARK_DIR = convert_path(get_full_path_to_project_root_dir() + "/data/samples/landmark/");
const std::string PATH_TO_RGB_DIR = convert_path(get_full_path_to_project_root_dir() + "/data/samples/rgb/");
const std::string PATH_TO_DEPTH_DIR = convert_path(get_full_path_to_project_root_dir() + "/data/samples/depth/");
const std::string PATH_TO_MESH_DIR = convert_path(get_full_path_to_project_root_dir() + "/data/outputMesh/");

const std::map<std::string, std::string> FACE_MODEL_TO_DIR_MAP = {
	{ "BFM17", convert_path(get_full_path_to_project_root_dir() + "/data/BFM17.h5")},
};
const std::map<std::string, std::string> FACE_MODEL_TO_LM_DIR_MAP = {
	{ "BFM17", convert_path(get_full_path_to_project_root_dir() + "/data/BFM17_68_Landmarks.txt")},
};
// h5 hierarchy path
const std::map<std::pair<std::string, std::string>, std::string> H5_PATH_MAP = {
	// BFM 2017
	{ std::make_pair("shape", "triangulation"), "/shape/representer/cells"},	// Triangulation (identical for shape/expr/color)
	{ std::make_pair("shape", "basis"), "/shape/model/pcaBasis"},
	{ std::make_pair("shape", "mean"), "/shape/model/mean"},
	{ std::make_pair("shape", "variance"), "/shape/model/pcaVariance"},
	{ std::make_pair("expression", "basis"), "/expression/model/pcaBasis"},
	{ std::make_pair("expression", "mean"), "/expression/model/mean"},
	{ std::make_pair("expression", "variance"), "/expression/model/pcaVariance"},
	{ std::make_pair("color", "basis"), "/color/model/pcaBasis"},
	{ std::make_pair("color", "variance"), "/color/model/pcaVariance"},
	{ std::make_pair("color", "mean"), "/color/model/mean"},
};


class DataHandler {
public:
	// read the precomputed landmarks from the file 
	static void loadLandmarks(std::string fileName, MatrixX2d& landmarks) {

		landmarks = MatrixXd(NUM_LANDMARKS, LANDMARK_DIM);
		std::string pathToFile = PATH_TO_LANDMARK_DIR + fileName + ".txt";
		std::ifstream f(pathToFile);
		if (!f.is_open()) std::cerr << "failed to open: " << pathToFile << std::endl;
		for (unsigned int i = 0; i < NUM_LANDMARKS; i++) {
			for (unsigned int j = 0; j < LANDMARK_DIM; j++) {
				f >> landmarks(i, j);
			}
		}
	}
	// read rgb value of the pixels from the image
	static void loadRGB(std::string fileName, std::vector<MatrixXd>& rgb) {
		std::string pathToFile = PATH_TO_RGB_DIR + fileName + ".jpeg";
		try
		{
			cv::Mat image = cv::imread(pathToFile, cv::IMREAD_COLOR);
			cv::Mat rgbMat[3];
			split(image, rgbMat);	//split source
			MatrixXd r, g, b;
			cv::cv2eigen(rgbMat[0], r);
			rgb[0] = r;
			cv::cv2eigen(rgbMat[1], g);
			rgb[1] = g;
			cv::cv2eigen(rgbMat[2], b);
			rgb[2] = b;
		}
		catch (cv::Exception& e)
		{
			std::cerr << "cv2 exception reading: " << pathToFile << std::endl;
			std::cerr << e.what() << std::endl;
		}
	}
	// read the depth map of the image
	static void loadDepthMap(std::string fileName, MatrixXd& depthMap) {
		std::string pathToFile = PATH_TO_DEPTH_DIR + fileName + ".jpeg";
		try
		{
			cv::Mat image = cv::imread(pathToFile, cv::IMREAD_GRAYSCALE);
			cv::cv2eigen(image, depthMap);
		}
		catch (cv::Exception& e)
		{
			std::cerr << "cv2 exception reading: " << pathToFile << std::endl;
			std::cerr << e.what() << std::endl;
		}
	}
	// read basis from hdf5 file
	static void readBasis(std::string faceModelName, std::string basisName, MatrixXf& basis) {
		hid_t h5file = H5Fopen(FACE_MODEL_TO_DIR_MAP.at(faceModelName).c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
		hid_t h5d = H5Dopen2(h5file, H5_PATH_MAP.at(std::make_pair(basisName, "basis")).c_str(), H5P_DEFAULT);
		if (h5d < 0) std::cerr << "Error reading basis from: " << faceModelName << std::endl;
		else {
			std::vector<unsigned int> shape = get_h5_dataset_shape(h5d);
			basis = MatrixXd(shape[1], shape[0]);
			H5Dread(h5d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &basis(0));
			basis.transposeInPlace();
		}
		H5Dclose(h5d);
		H5Fclose(h5file);

	}
	// read mean from hdf5 file
	static void readMean(std::string faceModelName, std::string meanName, VectorXf& mean) {
		hid_t h5file = H5Fopen(FACE_MODEL_TO_DIR_MAP.at(faceModelName).c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
		hid_t h5d = H5Dopen2(h5file, H5_PATH_MAP.at(std::make_pair(meanName, "mean")).c_str(), H5P_DEFAULT);
		if (h5d < 0) std::cerr << "Error reading mean from: " << faceModelName << std::endl;
		else {
			mean = VectorXd(get_h5_dataset_shape(h5d)[0]);
			H5Dread(h5d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &mean(0));
		}
		H5Dclose(h5d);
		H5Fclose(h5file);
	}
	// read variance from hdf5 file
	static void readVariance(std::string faceModelName, std::string varianceName, VectorXf& variance) {
		hid_t h5file = H5Fopen(FACE_MODEL_TO_DIR_MAP.at(faceModelName).c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
		hid_t h5d = H5Dopen2(h5file, H5_PATH_MAP.at(std::make_pair(varianceName, "variance")).c_str(), H5P_DEFAULT);
		if (h5d < 0) std::cerr << "Error reading variance from: " << faceModelName << std::endl;
		else {
			variance = VectorXd(get_h5_dataset_shape(h5d)[0]);
			H5Dread(h5d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &variance(0));
		}
		H5Dclose(h5d);
		H5Fclose(h5file);
	}
	// read triangulation from hdf5 file
	static void readTriangulation(std::string faceModelName, MatrixX3i& triangulation) {
		hid_t h5file = H5Fopen(FACE_MODEL_TO_DIR_MAP.at(faceModelName).c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
		hid_t h5d = H5Dopen2(h5file, H5_PATH_MAP.at(std::make_pair("shape", "triangulation")).c_str(), H5P_DEFAULT);
		if (h5d < 0) std::cerr << "Error reading triangulation from: " << faceModelName << std::endl;
		else {
			std::vector<unsigned int> shape = get_h5_dataset_shape(h5d);
			triangulation =  MatrixXi(shape[1], shape[0]);
			H5Dread(h5d, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &triangulation(0));
		}
		H5Dclose(h5d);
		H5Fclose(h5file);
	}
	// read 68 facial landmarks (vertex index) of the corresponding face model
	static void readFaceModelLandmarks(std::string faceModelName, VectorXi& landmarks) {
		std::ifstream f(FACE_MODEL_TO_LM_DIR_MAP.at(faceModelName));
		if (!f.is_open()) std::cerr << "failed to open: " << FACE_MODEL_TO_LM_DIR_MAP.at(faceModelName) << std::endl;
		landmarks = VectorXi(NUM_LANDMARKS);
		int i = 0;
		for (int i = 0; i < NUM_LANDMARKS; i++) f >> landmarks(i);
		f.close();
	}
	static bool writeMesh(const Mesh& mesh, const std::string& filename) {
		std::string pathToFile = PATH_TO_MESH_DIR + filename + ".off";

		//number of valid vertices
		unsigned int nVertices = 0;
		nVertices = mesh.vertices.rows();

		//number of valid faces
		unsigned nFaces = 0;
		nFaces = mesh.faces.rows();

		// Write off file
		std::ofstream outFile(pathToFile);
		if (!outFile.is_open()) return false;

		// write header
		outFile << "COFF" << std::endl;

		outFile << "# numVertices numFaces numEdges" << std::endl;

		outFile << nVertices << " " << nFaces << " " << 0 << std::endl;

		// save vertices
		outFile << "# list of vertices" << std::endl;
		outFile << "# X Y Z R G B A" << std::endl;
		for (unsigned i = 0; i < nVertices; ++i) {
			outFile << mesh.vertices.row(i) << " ";
			for (int j = 0; j < 3; ++j) {
				if (mesh.colors(i, j) < 0)
					outFile << "0";
				if (mesh.colors(i, j) > 255)
					outFile << "255";
				else
					outFile << mesh.colors(i, j);
				outFile << " ";
			}
			outFile << "255" << std::endl;
		}

		//save valid faces
		outFile << "# list of faces" << std::endl;
		outFile << "# nVerticesPerFace idx0 idx1 idx2 ..." << std::endl;

		for (unsigned i = 0; i < nFaces; ++i) {
			outFile << "3 " << mesh.faces.row(i) << std::endl;
		}
		return true;
	}

};
	




