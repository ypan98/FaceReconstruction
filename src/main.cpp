#include <iostream>
#include "Optimizer.h"
#include "Face.h"
#include "Renderer.h"
#include <chrono>
#include <omp.h>
#include <DataHandler.h>
#include <Utils.h>
using namespace std;

vector<string> taskOptions{ "Face reconstruction of single image", "Expression transfer of images", "Expression transfer of sequences" };
int taskOption = -1;

// Allows user to select an option from predefined task list
void handleMenu() {
	cout << "Please select a task:\n";
	for (int i = 0; i < taskOptions.size(); i++) cout << i + 1 << ". " << taskOptions[i] << endl;
	while (cin >> taskOption) {
		if (taskOption > 0 && taskOption <= taskOptions.size()) break;
		else cout << "Enter a valid option\n";
	}
}

void performTask1() {
	// initialize
	Face sourceFace = Face("W00151", "BFM17");
	Image img = sourceFace.getImage();
	Optimizer optimizer(sourceFace, false);
	// optimize params
	optimizer.optimize(false, true);
	// render the result
	Matrix4f mvp_matrix = sourceFace.getFullProjectionMatrix().transpose().cast<float>();
	Matrix4f mv_matrix = sourceFace.getExtrinsics().transpose().cast<float>();
	VectorXf vertices = sourceFace.getShapeWithExpression(sourceFace.getGamma()).cast<float>();
	VectorXf colors = sourceFace.getColor().cast<float>();
	VectorXf sh_red_coefficients = sourceFace.getSHRedCoefficients().cast<float>();
	VectorXf sh_green_coefficients = sourceFace.getSHGreenCoefficients().cast<float>();
	VectorXf sh_blue_coefficients = sourceFace.getSHBlueCoefficients().cast<float>();
	Renderer rendererOriginal(sourceFace.getFaceModel(), img.getHeight(), img.getWidth());
	rendererOriginal.render(mvp_matrix, mv_matrix, vertices, colors, sh_red_coefficients, sh_green_coefficients, sh_blue_coefficients, sourceFace.get_z_near(),
		sourceFace.get_z_far());
	sourceFace.setColor(rendererOriginal.get_re_rendered_vertex_color().cast<double>());
	imshow("Reconstructed face", rendererOriginal.get_color_buffer());
	cv::waitKey(0);
	// write out mesh
	sourceFace.writeReconstructedFace();
	cout << "Resulting mesh is saved in /data/outputMesh/" << endl;
}

void performTask2() {
	// source face fitting
	Face sourceFace = Face("X00081", "BFM17");
	Image img = sourceFace.getImage();
	Optimizer optimizer(sourceFace);
	optimizer.optimize(false, false);
	// target face fitting
	Face targetFace = Face("W00001", "BFM17");
	Image img2 = targetFace.getImage();
	Optimizer optimizer2(targetFace);
	optimizer2.optimize(false, false);
	// transfer expression
	targetFace.setGamma(sourceFace.getGamma());
	// render the result
	FaceModel faceModelAux = targetFace.getFaceModel();
	Matrix4f mvp_matrix = targetFace.getFullProjectionMatrix().transpose().cast<float>();
	Matrix4f mv_matrix = targetFace.getExtrinsics().transpose().cast<float>();
	VectorXf vertices = targetFace.getShapeWithExpression(targetFace.getGamma()).cast<float>();
	VectorXf colors = targetFace.getColor().cast<float>();
	MatrixX3i triangulation = faceModelAux.getTriangulation();
	VectorXf sh_red_coefficients = targetFace.getSHRedCoefficients().cast<float>();
	VectorXf sh_green_coefficients = targetFace.getSHGreenCoefficients().cast<float>();
	VectorXf sh_blue_coefficients = targetFace.getSHBlueCoefficients().cast<float>();
	// adding square behing the mouse
	addBoundingSquareBehindMouse(vertices, colors, triangulation, faceModelAux.getLandmarks());
	faceModelAux.setTriangulation(triangulation);
	faceModelAux.setExtraVertices(4);
	Renderer renderer(faceModelAux, img2.getHeight(), img2.getWidth());
	renderer.render(mvp_matrix, mv_matrix, vertices, colors, sh_red_coefficients, sh_green_coefficients, sh_blue_coefficients, targetFace.get_z_near(),
		targetFace.get_z_far());
	cv::Mat renderedImg = renderer.get_color_buffer();
	// merging the original image's backround to the rendered image
	mergeBackground(targetFace.getImage().getBGRCopy(), renderedImg);
	targetFace.setColor(renderer.get_re_rendered_vertex_color().cast<double>());
	imshow("Expression transferred target face", renderer.get_color_buffer());
	cv::waitKey(0);
	// write out mesh
	targetFace.writeReconstructedFace();
	cout << "Resulting mesh is saved in /data/outputMesh/" << endl;
	renderer.clear_buffers();
	renderer.terminate_rendering_context();
}

/* Expression transfer of a sequence of images (target and source).
* Both sequence need to have same length.
* The expression is transferred from a frame of source to the coreesponding frame of the target sequence.
* First frame is considered neutral, therefore we use it to fix the base shape and color (alpha, beta). In the following frames we only optimize for
* expression (gamma) and illumination (sh coefficients)
* Note: this function is design for the way we name the sequence images
*/
void performTask3() {
	string sourceActor = "Z";
	string targetActor = "W";
	int frameStep = 1;
	int startFrame = 1;
	int endFrame = 224;
	Face sourceFace = Face(sourceActor+"00000", "BFM17");
	Face targetFace = Face(targetActor+"00000", "BFM17");
	Image img2 = targetFace.getImage();
	Optimizer optimizer(sourceFace, false);
	optimizer.optimize(false, false,false);
	Optimizer optimizer2(targetFace, false);
	optimizer2.optimize(false, false,false);
	VectorXd sourceNeutralGamma = sourceFace.getGamma();
	VectorXd targetNeutralGamma = targetFace.getGamma();

	FaceModel faceModelAux = targetFace.getFaceModel();
	VectorXf vertices = targetFace.getShapeWithExpression(targetFace.getGamma()).cast<float>();
	VectorXf colors = targetFace.getColor().cast<float>();
	MatrixX3i triangulation = faceModelAux.getTriangulation();
	addBoundingSquareBehindMouse(vertices, colors, triangulation, faceModelAux.getLandmarks());
	faceModelAux.setTriangulation(triangulation);
	faceModelAux.setExtraVertices(4);
	Renderer renderer(faceModelAux, img2.getHeight(), img2.getWidth());

	for (int i = startFrame; i <= endFrame; i += frameStep) {
		Matrix4d extrinsics = Matrix4d::Identity();
		extrinsics(2, 3) = -0.6;
		sourceFace.setExtrinsics(extrinsics);
		targetFace.setExtrinsics(extrinsics);

		cout << "Current frame: " << i << endl;
		string frameName = to_string(i);
		if (i >= 100) frameName = "00" + frameName;
		else if (i >= 10) frameName = "000" + frameName;
		else if (i >= 1) frameName = "0000" + frameName;
		sourceFace.setImage(sourceActor + frameName);
		targetFace.setImage(targetActor + frameName);
		optimizer.optimize(true, false, false);
		optimizer2.optimize(true, false, false);
		//targetFace.setGamma(targetNeutralGamma + sourceFace.getGamma()-sourceNeutralGamma);
		// render the result
		Matrix4f mvp_matrix = targetFace.getFullProjectionMatrix().transpose().cast<float>();
		Matrix4f mv_matrix = targetFace.getExtrinsics().transpose().cast<float>();
		// transfer expression
		VectorXd transferred_gamma = targetNeutralGamma + sourceFace.getGamma() - sourceNeutralGamma;
		VectorXf vertices = targetFace.getShapeWithExpression(transferred_gamma).cast<float>();
		VectorXf colors = targetFace.getColor().cast<float>();
		MatrixX3i triangulation = faceModelAux.getTriangulation();
		VectorXf sh_red_coefficients = targetFace.getSHRedCoefficients().cast<float>();
		VectorXf sh_green_coefficients = targetFace.getSHGreenCoefficients().cast<float>();
		VectorXf sh_blue_coefficients = targetFace.getSHBlueCoefficients().cast<float>();
		// adding square behing the mouse
		addBoundingSquareBehindMouse(vertices, colors, triangulation, faceModelAux.getLandmarks());
		renderer.render(mvp_matrix, mv_matrix, vertices, colors, sh_red_coefficients, sh_green_coefficients, sh_blue_coefficients, targetFace.get_z_near(),
			targetFace.get_z_far());
		cv::Mat renderedImg = renderer.get_color_buffer();
		cv::Mat originalCopy = targetFace.getImage().getBGRCopy();
		// merging the original image's backround to the rendered image
		mergeBackground(originalCopy, renderedImg);
		DataHandler::saveFrame(originalCopy, targetActor + frameName);
		renderer.clear_buffers();
	}
	cout << "Resulting sequence saved in /data/outputSequence/" << endl;
}

// Perform the selected task
void performTask() {
	switch (taskOption) {
		// Reconstruct face mesh from a sample image
		case 1:
		{
			performTask1();
			break;
		}
		// Face reconstruction of two images and transfer the expression from source to target face 
		case 2:
		{
			performTask2();
			break;
		}
		// Expression transfer in a sequence of images
		case 3:
		{
			performTask3();
			break;
		}
		default: {
			cout << "Invalid task" << endl;
			break;
		}
	}
}


int main(int argc, char** argv) {
	omp_set_num_threads(omp_get_max_threads());
	google::InitGoogleLogging(argv[0]);
	omp_set_num_threads(omp_get_max_threads());
	handleMenu();
	performTask();
	return 0;
}