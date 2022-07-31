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
		Face sourceFace = Face("sample1", "BFM17");
		Image img = sourceFace.getImage();
		Renderer rendererOriginal(sourceFace.getFaceModel(), img.getHeight(), img.getWidth());
		Renderer rendererDownsampled(sourceFace.getFaceModel(), img.getHeightDown(), img.getWidthDown());
		sourceFace.setIntrinsics(double(60), double(sourceFace.getImage().getWidth()) / double(sourceFace.getImage().getHeight()),
			double(8800), double(9000));
		Optimizer optimizer(sourceFace);
		// optimize params
		optimizer.optimize(0);
		// render the result
		Matrix4f mvp_matrix = sourceFace.getFullProjectionMatrix().transpose().cast<float>();
		Matrix4f mv_matrix = sourceFace.getExtrinsics().transpose().cast<float>();
		VectorXf vertices = sourceFace.getShape().cast<float>();
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
		cout << "Resulting mesh is saved in ./data" << endl;
		break;
	}
	case 2:
	{
		// reconstruct source face
		//Face sourceFace = Face("sample1", "BFM17");
		//render.initialiaze_rendering_context(sourceFace.getFaceModel(), sourceFace.getImage().getHeight(), sourceFace.getImage().getWidth());
		//sourceFace.setIntrinsics(double(60), double(sourceFace.getImage().getWidth()) / double(sourceFace.getImage().getHeight()),
		//	double(8800), double(9000));
		//optimizer.optimize(sourceFace, 0);
		//// reconstruct target face
		//Face targetFace = Face("sample2", "BFM17");
		//render.initialiaze_rendering_context(sourceFace.getFaceModel(), sourceFace.getImage().getHeight(), sourceFace.getImage().getWidth());
		//targetFace.setIntrinsics(double(60), double(sourceFace.getImage().getWidth()) / double(sourceFace.getImage().getHeight()),
		//	double(8800), double(9000));
		//optimizer.optimize(targetFace, 0);
		//// transfer expression and write out mesh
		//targetFace.transferExpression(sourceFace);
		//targetFace.writeReconstructedFace();
		//break;
	}
	case 3:
	{

	}
	default: {
		cout << "Invalid task" << endl;
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