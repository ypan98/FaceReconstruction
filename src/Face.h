#pragma once
#include "Image.h"
#include "FaceModel.h"
#include "Utils.h"

// Class for a reconstructed 3D face
class Face {
public:
	Face(std::string _imageFileName = "sample1", std::string _faceModel = "BFM17") {
		image = Image(_imageFileName);
		faceModel = FaceModel(_faceModel);		
		alpha = VectorXd::Zero(faceModel.getAlphaSize());
		beta = VectorXd::Zero(faceModel.getBetaSize());
		gamma = VectorXd::Zero(faceModel.getGammaSize());
		sh_red_coefficients = VectorXd::Zero(9);
		sh_green_coefficients = VectorXd::Zero(9);
		sh_blue_coefficients = VectorXd::Zero(9);
		shape = faceModel.getShapeMean();
		color = faceModel.getColorMean();
		intrinsics = Matrix4d::Identity();
		extrinsics = Matrix4d::Identity();
		extrinsics(2, 3) = -400;
	}

	// Calls DataHandler to write the recontructed mesh in .obj format
	void writeReconstructedFace() {
		Mesh mesh = toMesh();
		DataHandler::writeMesh(mesh, image.getFileName());
	}
	
	// Transfer the expression from the source face to the current face
	void transferExpression(const Face& sourceFace) {
		this->gamma = sourceFace.gamma;
	}

	unsigned getNumTriangles() const {
		return faceModel.getTriangulationRows();
	}

	unsigned getNumVertices() const {
		return faceModel.getNumVertices();
	}

	// Randomize parameters (for testing purpose)
	void randomizeParameters(double scaleAlpha = 1, double scaleBeta = 1, double scaleGamma = 1) {
		alpha = VectorXd::Random(faceModel.getAlphaSize()) * scaleAlpha;
		beta = VectorXd::Random(faceModel.getBetaSize()) * scaleBeta;
		gamma = VectorXd::Random(faceModel.getGammaSize()) * scaleGamma;
	}

	// construct the mesh with alpha, beta, gamma and face model variables
	Mesh toMesh() {
		Mesh mesh;
		MatrixXd vertices = shape;
		vertices.resize(3, faceModel.getNumVertices());
		mesh.vertices = vertices.transpose();

		MatrixXd colors = color;
		colors.resize(3, faceModel.getNumVertices());
		mesh.colors = colors.transpose();

		mesh.faces = faceModel.getTriangulation();
		return mesh;
	}

	// getters and setters

	void setAlpha(VectorXd _alpha) {
		alpha = _alpha;
	}
	VectorXd getAlpha() {
		return alpha;
	}
	void setBeta(VectorXd _beta) {
		beta = _beta;
	}
	VectorXd getBeta() {
		return beta;
	}
	void setGamma(VectorXd _gamma) {
		gamma = _gamma;
	}
	VectorXd getGamma() {
		return gamma;
	}
	void setSHRedCoefficients(VectorXd _sh_red_coefficients) {
		sh_red_coefficients = _sh_red_coefficients;
	}
	VectorXd getSHRedCoefficients() {
		return sh_red_coefficients;
	}
	void setSHGreenCoefficients(VectorXd _sh_green_coefficients) {
		sh_green_coefficients = _sh_green_coefficients;
	}
	VectorXd getSHGreenCoefficients() {
		return sh_green_coefficients;
	}
	void setSHBlueCoefficients(VectorXd _sh_blue_coefficients) {
		sh_blue_coefficients = _sh_blue_coefficients;
	}
	VectorXd getSHBlueCoefficients() {
		return sh_blue_coefficients;
	}
	void setIntrinsics(Matrix4d _intrinsics) {
		intrinsics = _intrinsics;
	}
	Matrix4d getIntrinsics() {
		return intrinsics;
	}
	void setExtrinsics(Matrix4d _extrinsics) {
		extrinsics = _extrinsics;
	}
	Matrix4d getExtrinsics() {
		return extrinsics;
	}
	Matrix4d getFullProjectionMatrix() {
		return intrinsics * extrinsics;
	}

	Image getImage() {
		return image;
	}
	FaceModel getFaceModel() {
		return faceModel;
	}
	// geometry
	MatrixX3d calculateVertices() {
		MatrixXd vertices = faceModel.getShapeMean() + faceModel.getShapeBasis() * alpha +
			faceModel.getExpMean() + faceModel.getExpBasis() * gamma;
		vertices.resize(3, faceModel.getNumVertices());
		return vertices.transpose();
	}
	// color
	MatrixX3d calculateColors() {
		MatrixXd colors = faceModel.getColorMean() + faceModel.getColorBasis() * beta;
		colors.resize(3, faceModel.getNumVertices());
		return colors.transpose();
	}
	VectorXd calculateVerticesDefault() {
		return faceModel.getShapeMean() + faceModel.getShapeBasis() * alpha +
			faceModel.getExpMean() + faceModel.getExpBasis() * gamma;
	}
	VectorXd calculateColorsDefault() {
		return faceModel.getColorMean() + faceModel.getColorBasis() * beta;
	}
	VectorXd getShapeBlock(unsigned startRow, unsigned numRows) const {
		return shape.block(startRow, 0, numRows, 1);
	}
	VectorXd getColorBlock(unsigned startRow, unsigned numRows) const {
		return color.block(startRow, 0, numRows, 1);
	}
	void setShape(VectorXd shape_) {
		shape = shape_;
	}
	VectorXd getShape() const {
		return shape;
	}
	void setColor(VectorXd color_) {
		color = color_;
	}
	VectorXd getColor() const {
		return color;
	}
private:
	VectorXd alpha, beta, gamma, sh_red_coefficients, sh_green_coefficients, sh_blue_coefficients;	// parameters to optimize
	VectorXd shape, color;
	Matrix4d intrinsics;	// given by camera manufacturer, otherwise hardcode it?
	Matrix4d extrinsics;	// given by optimization
	Image image;	// the corresponding image
	FaceModel faceModel;	// the used face model, ie BFM17
};