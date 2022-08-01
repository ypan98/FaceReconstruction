#include <iostream>
#include "Optimizer.h"
#include "Face.h"
#include "Renderer.h"
#include <chrono>
#include <omp.h>
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

// Perform the selected task
void performTask() {
	switch (taskOption) {
		// Reconstruct face mesh from a sample image
		case 1:
		{
			// initialize
			Face sourceFace = Face("00100", "BFM17");
			Image img = sourceFace.getImage();
			Renderer rendererDownsampled(sourceFace.getFaceModel(), img.getHeightDown(), img.getWidthDown());
			Optimizer optimizer(sourceFace);
			// optimize params
			optimizer.optimize(false, false);
			// render the result
			Matrix4f mvp_matrix = sourceFace.getFullProjectionMatrix().transpose().cast<float>();
			Matrix4f mv_matrix = sourceFace.getExtrinsics().transpose().cast<float>();
			VectorXf vertices = sourceFace.getShapeWithExpression().cast<float>();
			VectorXf colors = sourceFace.getColor().cast<float>();
			VectorXf sh_red_coefficients = sourceFace.getSHRedCoefficients().cast<float>();
			VectorXf sh_green_coefficients = sourceFace.getSHGreenCoefficients().cast<float>();
			VectorXf sh_blue_coefficients = sourceFace.getSHBlueCoefficients().cast<float>();
			rendererDownsampled.render(mvp_matrix, mv_matrix, vertices, colors, sh_red_coefficients, sh_green_coefficients, sh_blue_coefficients, sourceFace.get_z_near(),
				sourceFace.get_z_far());
			sourceFace.setColor(rendererDownsampled.get_re_rendered_vertex_color().cast<double>());
			imshow("Reconstructed face", rendererDownsampled.get_color_buffer());
			cv::waitKey(0);
			// write out mesh
			sourceFace.writeReconstructedFace();
			cout << "Resulting mesh is saved in /data/outputMesh/" << endl;
			break;
		}
		// Face reconstruction of two images and transfer the expression from source to target face 
		case 2:
		{
			// source face fitting
			Face sourceFace = Face("sample6", "BFM17");
			Image img = sourceFace.getImage();
			Optimizer optimizer(sourceFace);
			optimizer.optimize(false, false);
			// target face fitting
			Face targetFace = Face("sample2", "BFM17");
			Image img2 = targetFace.getImage();
			Optimizer optimizer2(targetFace);
			optimizer2.optimize(false, false);
			// transfer expression
			 targetFace.transferExpression(sourceFace);
			// render the result
			Matrix4f mvp_matrix = targetFace.getFullProjectionMatrix().transpose().cast<float>();
			Matrix4f mv_matrix = targetFace.getExtrinsics().transpose().cast<float>();
			VectorXf vertices = targetFace.getShapeWithExpression().cast<float>();
			VectorXf colors = targetFace.getColor().cast<float>();
			VectorXf sh_red_coefficients = targetFace.getSHRedCoefficients().cast<float>();
			VectorXf sh_green_coefficients = targetFace.getSHGreenCoefficients().cast<float>();
			VectorXf sh_blue_coefficients = targetFace.getSHBlueCoefficients().cast<float>();
			Renderer rendererDownsampled(targetFace.getFaceModel(), img2.getHeightDown(), img2.getWidthDown());
			rendererDownsampled.render(mvp_matrix, mv_matrix, vertices, colors, sh_red_coefficients, sh_green_coefficients, sh_blue_coefficients, targetFace.get_z_near(),
				targetFace.get_z_far());
			targetFace.setColor(rendererDownsampled.get_re_rendered_vertex_color().cast<double>());
			imshow("Expression transferred target face", rendererDownsampled.get_color_buffer());
			cv::waitKey(0);
			// write out mesh
			targetFace.writeReconstructedFace();
			cout << "Resulting mesh is saved in /data/outputMesh/" << endl;
			break;
		}
		case 3:
		{
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