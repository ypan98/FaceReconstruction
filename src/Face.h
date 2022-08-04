#pragma once
#include "Image.h"
#include "FaceModel.h"
#include "Utils.h"
#include <math.h>

// Class for a reconstructed 3D face
class Face {
public:
	// constructor to initialize with the specified image and face model
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
		setIntrinsics(double(54), double(image.getWidth()) / double(image.getHeight()),
			double(0.01), double(10));
		extrinsics = Matrix4d::Identity();
		extrinsics(2, 3) = -0.6;
	}
	// Calls DataHandler to write the recontructed mesh in .obj format
	void writeReconstructedFace() {
		Mesh mesh = toMesh();
		DataHandler::writeMesh(mesh, image.getFileName());
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
		//MatrixXd vertices = shape + faceModel.getExpMean() + faceModel.getExpBasis() * gamma;
		MatrixXd vertices = shape + faceModel.getExpMean() + faceModel.getExpBasis() * gamma;
		vertices.resize(3, faceModel.getNumVertices());
		mesh.vertices = vertices.transpose();

		MatrixXd colors = color;
		colors.resize(3, faceModel.getNumVertices());
		mesh.colors = colors.transpose();

		mesh.faces = faceModel.getTriangulation();
		return mesh;
	}
	// geometry
	MatrixX3d calculateVertices() {
		MatrixXd vertices = faceModel.getShapeMean() + faceModel.getShapeBasis() * alpha +
			faceModel.getExpMean() + faceModel.getExpBasis() * gamma;
		vertices.resize(3, faceModel.getNumVertices());
		return vertices.transpose();
	}
	VectorXd calculateVerticesDefault() {
		return faceModel.getShapeMean() + faceModel.getShapeBasis() * alpha +
			faceModel.getExpMean() + faceModel.getExpBasis() * gamma;
	}
	VectorXd calculateVerticesNeutralExp() {
		return faceModel.getShapeMean() + faceModel.getShapeBasis() * alpha;
	}

	// color
	MatrixX3d calculateColors() {
		MatrixXd colors = faceModel.getColorMean() + faceModel.getColorBasis() * beta;
		colors.resize(3, faceModel.getNumVertices());
		return colors.transpose();
	}
	VectorXd calculateColorsDefault() {
		return faceModel.getColorMean() + faceModel.getColorBasis() * beta;
	}

	// getters and setters
	void setImage(std::string _imageFileName) {
		this->image = Image(_imageFileName);
	}
	unsigned getNumTriangles() const {
		return faceModel.getTriangulationRows();
	}
	unsigned getNumVertices() const {
		return faceModel.getNumVertices();
	}
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
	void setIntrinsics(double fov_, double aspect_ratio_, double z_near_, double z_far_) {
		double f = 1. / tan(fov_ * (3.1415926 / 360));
		Matrix4d perspective_projection_matrix = Matrix4d::Zero();
		perspective_projection_matrix(0, 0) = f / aspect_ratio_;
		perspective_projection_matrix(1, 1) = f;
		perspective_projection_matrix(2, 2) = (z_near_ + z_far_) / (z_near_ - z_far_);
		perspective_projection_matrix(2, 3) = (2 * z_near_ * z_far_) / (z_near_ - z_far_);
		perspective_projection_matrix(3, 2) = -1;
		fov = fov_;
		z_near = z_near_;
		z_far = z_far_;
		intrinsics = perspective_projection_matrix;
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
	VectorXd getShapeWithExpression(VectorXd& gamma_) const {
		return shape + faceModel.getExpMean() + faceModel.getExpBasis() * gamma_;
	}
	void setColor(VectorXd color_) {
		color = color_;
	}
	VectorXd getColor() const {
		return color;
	}
	float get_z_near() const {
		return z_near;
	}
	float get_z_far() const {
		return z_far;
	}
private:
	VectorXd alpha, beta, gamma, sh_red_coefficients, sh_green_coefficients, sh_blue_coefficients;	// parameters to optimize
	VectorXd shape, color;
	Matrix4d intrinsics;	// given by camera manufacture
	Matrix4d extrinsics;	// given by landamark fitting
	Image image;	// the corresponding image
	FaceModel faceModel;	// the used face model, ie BFM17
	double fov, z_near, z_far;	// rasterization params
};