#pragma once
#include "DataHandler.h"

#define BFM_ALPHA_SIZE 199
#define BFM_BETA_SIZE 199
#define BFM_GAMMA_SIZE 99

// Class for a face model
class FaceModel{

public:
	FaceModel() {
	}
	FaceModel(std::string _faceModel) {
		faceModelName = _faceModel;
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
		DataHandler::readFaceModelLandmarks(_faceModel, landmarks);
	}

	std::string getFaceModelName() {
		return faceModelName;
	}
	unsigned int getNumVertices() const {
		return shapeBasis.rows()/3;
	}

	unsigned int getAlphaSize() const {
		return shapeBasis.cols();
	}

	unsigned int getBetaSize() const {
		return colorBasis.cols();
	}

	unsigned int getGammaSize() const {
		return expBasis.cols();
	}

	VectorXd getShapeMean() const {
		return shapeMean;
	}

	double getShapeMeanElem(unsigned idx) const {
		return shapeMean(idx, 0);
	}

	VectorXd getColorMean() const {
		return colorMean;
	}

	double getColorMeanElem(unsigned idx) const {
		return colorMean(idx, 0);
	}

	VectorXd getExpMean() const {
		return expMean;
	}

	double getExpMeanElem(unsigned idx) const {
		return expMean(idx, 0);
	}

	VectorXd getShapeVar() const {
		return shapeVar;
	}

	VectorXd getColorVar() const {
		return colorVar;
	}

	VectorXd getExpVar() const {
		return expVar;
	}

	MatrixXd getShapeBasis() const {
		return shapeBasis;
	}

	MatrixXd getShapeBasisStdMultiplied() const {
		return (shapeVar.cwiseSqrt().asDiagonal() * shapeBasis.transpose()).transpose();
	}

	double getShapeBasisStdMultipliedElem(unsigned i, unsigned j) const {
		return shapeBasis(i, j) * sqrt(shapeVar(j, 0));
	}

	MatrixXd getColorBasis() const {
		return colorBasis;
	}

	MatrixXd getColoBasisStdMultiplied() const {
		return (colorVar.cwiseSqrt().asDiagonal() * colorBasis.transpose()).transpose();
	}

	double getColoBasisStdMultipliedElem(unsigned i, unsigned j) const {
		return colorBasis(i, j) * sqrt(colorVar(j, 0));
	}

	MatrixXd getExpBasis() const {
		return expBasis;
	}

	MatrixXd getExpBasisStdMultiplied() const {
		return (expVar.cwiseSqrt().asDiagonal() * expBasis.transpose()).transpose();
	}

	double getExpBasisStdMultipliedElem(unsigned i, unsigned j) const {
		return expBasis(i, j) * sqrt(expVar(j, 0));
	}

	MatrixX3i getTriangulation() const {
		return triangulation;
	}

	unsigned getNumLandmarks() const {
		return landmarks.rows();
	}

	VectorXi getLandmarks() const {
		return landmarks;
	}

	unsigned getLandmarkVertexIdx(unsigned i) const {
		return landmarks.row(i).value();
	}

//private:
  std::string faceModelName;
  // Mean of the eigenvectors in the basis. Shape = 3*num_vertices (because each vertex is a 3D point)
  VectorXd shapeMean, colorMean, expMean;
  // Variance of the eigenvectors in the basis. Shape = num_eigenvectors
  VectorXd shapeVar, colorVar, expVar;
  // Basis formed by the eigvectors. Shape = [3*num_vectices, num_eigenvectors]
  MatrixXd shapeBasis, colorBasis, expBasis;
  // triangulation of the vertices. Shape = [num_faces, 3]
  MatrixX3i triangulation;
  // vector with index of the vertices corresponding to the facial landmarks. Shape = [68, 1]
  VectorXi landmarks;
};