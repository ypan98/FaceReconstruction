#pragma once
#include "Eigen.h"
#include "DataHandler.h"

// Class for a face model
class FaceModel{

public:
	FaceModel(std::string _faceModel = "BFM17") {
		// TODO: call DataHandler functions to load and initialize class attributes
	}

	unsigned int getNumVertices() {
		return idMean.rows();
	}

	unsigned int getAlphaSize() {
		return idBasis.cols();
	}

	unsigned int getBetaSize() {
		return expBasis.cols();
	}

	unsigned int getGammaSize() {
		return albBasis.cols();
	}

private:
  
  // Mean of the eigenvectors in the basis. Shape = 3*num_vertices (each vertex is a 3D point)
  VectorXf idMean, albMean, expMean;
  // Basis formed by the eigvectors. Shape = [3*num_vectices, basis_size]
  MatrixXf idBasis, albBasis, expBasis;
  // triangulation of the vertices. Shape = [num_faces, 3]
  MatrixX3i triangulation;
};