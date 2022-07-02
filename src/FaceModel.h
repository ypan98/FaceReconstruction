#pragma once
#include "DataHandler.h"

// Class for a face model
class FaceModel{

public:
	FaceModel(std::string _faceModel = "BFM17") {
		DataHandler::readBasis(_faceModel, "shape", shapeBasis);
		DataHandler::readBasis(_faceModel, "color", colorBasis);
		DataHandler::readBasis(_faceModel, "expression", expBasis);
		DataHandler::readMean(_faceModel, "shape", shapeMean);
		DataHandler::readMean(_faceModel, "color", colorMean);
		DataHandler::readMean(_faceModel, "expression", expMean);
		DataHandler::readVariance(_faceModel, "shape", shapeVar);
		DataHandler::readVariance(_faceModel, "color", colorVar);
		DataHandler::readVariance(_faceModel, "expression", expVar);
		DataHandler::readTriangulation(_faceModel, triangulation);
	}

	unsigned int getNumVertices() {
		return shapeBasis.rows();
	}

	unsigned int getAlphaSize() {
		return shapeBasis.cols();
	}

	unsigned int getBetaSize() {
		return colorBasis.cols();
	}

	unsigned int getGammaSize() {
		return expBasis.cols();
	}

	VectorXf getShapeMean() {
		return shapeMean;
	}

	VectorXf getColorMean() {
		return colorMean;
	}

	VectorXf getExpMean() {
		return expMean;
	}

	VectorXf getShapeVar() {
		return shapeVar;
	}

	VectorXf getColorVar() {
		return colorVar;
	}

	VectorXf getExpVar() {
		return expVar;
	}

	MatrixXf getShapeBasis() {
		return shapeBasis;
	}

	MatrixXf getColorBasis() {
		return colorBasis;
	}

	MatrixXf getExpBasis() {
		return expBasis;
	}

	MatrixX3i getTriangulation() {
		return triangulation;
	}

private:
  
  // Mean of the eigenvectors in the basis. Shape = 3*num_vertices (because each vertex is a 3D point)
  VectorXf shapeMean, colorMean, expMean;
  // Variance of the eigenvectors in the basis. Shape = num_eigenvectors
  VectorXf shapeVar, colorVar, expVar;
  // Basis formed by the eigvectors. Shape = [3*num_vectices, num_eigenvectors]
  MatrixXf shapeBasis, colorBasis, expBasis;
  // triangulation of the vertices. Shape = [num_faces, 3]
  MatrixX3i triangulation;
};