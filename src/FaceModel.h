#pragma once

#include "Image.h"

struct Vertex
{
	// position stored as 4 floats (4th component is supposed to be 1.0)
	Vector4f position;
	// color stored as 4 unsigned char
	Vector4uc color;
};

class FaceModel{

//functions go here
public:
    //load data functions
    //3D to 2D projection functions



private:
  //optimized parameters alpha, beta, gama
  Eigen::Vector3d alpha;
  Eigen::Vector3d beta;
  Eigen::Vector3d gama;
  //basis, mean
  Eigen::Vector3d mean;
  Eigen::Vector3d basis;
  //intrinsics and extrinsice matrix(should be given by camera manufacturer)
  Eigen::Matrix3f depthIntrinsics;
  Eigen::Matrix4f depthExtrinsics;
  //landmarks
  std::vector<double> landmark;

};