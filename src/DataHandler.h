#pragma once

#include <fstream>
#include "Eigen.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <vector>
#include "Utils.h"

// Full paths
std::string PATH_TO_LANDMARK_DIR = convert_path(get_full_path_to_project_root_dir() + "/data/samples/landmark/");
std::string PATH_TO_RGB_DIR = convert_path(get_full_path_to_project_root_dir() + "/data/samples/rgb/");
std::string PATH_TO_DEPTH_DIR = convert_path(get_full_path_to_project_root_dir() + "/data/samples/depth/");
std::string PATH_TO_MESH_DIR = convert_path(get_full_path_to_project_root_dir() + "/data/outputMesh/");


unsigned int NUM_LANDMARKS = 68; // num of landmark points
unsigned int LANDMARK_DIM = 2; // each landmark is a 2D point

class DataHandler {
public:
	// read the precomputed landmarks from the file 
	static void loadLandmarks(std::string fileName, MatrixX2f& landmarks) {

		landmarks = MatrixXf(NUM_LANDMARKS, LANDMARK_DIM);
		std::string pathToFile = PATH_TO_LANDMARK_DIR + fileName + ".txt";
		std::ifstream f(pathToFile);
		if (!f.is_open())
			std::cerr << "failed to open: " << pathToFile << std::endl;
		for (unsigned int i = 0; i < NUM_LANDMARKS; i++) {
			for (unsigned int j = 0; j < LANDMARK_DIM; j++) {
				f >> landmarks(i, j);
			}
		}
	}
	// read rgb value of the pixels from the image
	static void loadRGB(std::string fileName, std::vector<MatrixXf>& rgb) {
		std::string pathToFile = PATH_TO_RGB_DIR + fileName + ".jpeg";
		try
		{
			cv::Mat image = cv::imread(pathToFile, cv::IMREAD_COLOR);
			cv::Mat rgbMat[3];
			split(image, rgbMat);	//split source
			MatrixXf r, g, b;
			cv::cv2eigen(rgbMat[0], r);
			rgb[0] = r;
			cv::cv2eigen(rgbMat[1], g);
			rgb[1] = g;
			cv::cv2eigen(rgbMat[2], b);
			rgb[2] = b;
		}
		catch (cv::Exception& e)
		{
			std::cout << "cv2 exception reading: " << pathToFile << std::endl;
			std::cout << e.what() << std::endl;
		}
	}
	// read the depth map of the image
	static void loadDepthMap(std::string fileName, MatrixXf& depthMap) {
		std::string pathToFile = PATH_TO_DEPTH_DIR + fileName + ".jpeg";
		try
		{
			cv::Mat image = cv::imread(pathToFile, cv::IMREAD_GRAYSCALE);
			cv::cv2eigen(image, depthMap);
		}
		catch (cv::Exception& e)
		{
			std::cout << "cv2 exception reading: " << pathToFile << std::endl;
			std::cout << e.what() << std::endl;
		}
	}
	bool writeMesh(const Mesh& mesh, const std::string& filename){
		std::string pathToFile = PATH_TO_MESH_DIR + filename + ".off";
	
		//number of valid vertices
		unsigned int nVertices = 0;
		nVertices = mesh.vertices.rows() * mesh.vertices.cols();

		//number of valid faces
		unsigned nFaces = 0;
		nFaces = mesh.triangles.rows(); 

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
		for(int i = 0; i < nVertices; ++i){
			outFile << mesh.vertices.row(i) << " ";
			for(int j = 0; j < 3; ++j){
				if(mesh.colors(i, j) < 0)
					outFile << "0";
				if(mesh.colors(i, j) > 255)
					outFile << "255";
				else
					outFile << mesh.colors(i, j);
			}
			outFile <<  "255" << std::endl;
		}
			
		//save valid faces
		outFile << "# list of faces" << std::endl;
		outFile << "# nVerticesPerFace idx0 idx1 idx2 ..." << std::endl;
		
		for(int i = 0; i < nFaces; ++i){
			outFile << 3 << " " << mesh.triangles.row(i) << std::endl;
		}	
	}
};
	




