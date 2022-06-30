#pragma once

#include <iostream>
#include <fstream>
//#include "Eigen.h"

#ifdef __unix__                   

#define OS_Windows 0

#elif    

#define OS_Windows 1

#endif

using namespace std;

string basePath = "D:\\TUM\\FaceReconstruction\\samples\\landmark\\";
int rows = 68; // num of landmark points
int cols = 2; // each landmark is a 2D point

class Landmark {
	public:
		/*static Matrix<float, rows, cols> get_land
		mark(string file_name) {
			Matrix<float, rows, cols> landmarks;
			ifstream file(basePath + file_name + ".txt");
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					f >> landmarks[i][j];
				}
			}
			return landmakrs;
 		}*/
		static void get_landmark(string file_name) {
			cout << OS_Windows << endl;
			/*string pathToFile = basePath + file_name + ".txt";
			ifstream f(pathToFile);
			cout << pathToFile << endl;
			if (!f.is_open())
				cout << "Error opening landmark file";
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					double d = -1;
					f >> d;
					cout << d << endl;

				}
			}*/
		}
};
