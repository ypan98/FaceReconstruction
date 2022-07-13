#include <iostream>
#include "Optimizer.h"
#include "Face.h"
#include "Renderer.h"
#include <chrono>
#include <omp.h>

using namespace std;

vector<string> taskOptions{ "Face reconstruction", "Expression transfer", "Rasterize random face" };
int taskOption = -1;
// For now we only hadle sample images case
// vector<string> inputOptions{ "Use sample image(s)" };
// int inputOption = -1;

Renderer Renderer::s_instance;

void handleMenu() {
	cout << "Please select a task:\n";
	for (int i = 0; i < taskOptions.size(); i++) cout << i + 1 << ". " << taskOptions[i] << endl;
	while (cin >> taskOption) {
		if (taskOption > 0 && taskOption <= taskOptions.size()) break;
		else cout << "Enter a valid option\n";
	}
	/*cout << "Please select an option for the input image(s):\n";
	for (int i = 0; i < inputOptions.size(); i++) cout << i+1 << ". " << inputOptions[i] << endl;
	while (cin >> inputOption) {
		if (inputOption >= 0 && inputOption <= inputOptions.size()) break;
		else cout << "Enter a valid option\n";
	}*/
}

void performTask() {
	Optimizer optimizer;
	switch (taskOption) {
	case 1:
	{
		// reconstruct face
		Face sourceFace = Face("sample1", "BFM17");
		// randomize for testing
		optimizer.optimize(sourceFace);
		// write out mesh
		sourceFace.writeReconstructedFace();
		cout << sourceFace.getAlpha() << endl;
		cout << sourceFace.getExtrinsics() << endl;
		cout << sourceFace.getIntrinsics() << endl;
		break;
	}
	case 2:
	{
		// reconstruct source face
		Face sourceFace = Face("sample1", "BFM17");
		optimizer.optimize(sourceFace);
		// reconstruct target face
		Face targetFace = Face("sample2", "BFM17");
		optimizer.optimize(targetFace);
		// transfer expression and write out mesh
		targetFace.transferExpression(sourceFace);
		targetFace.writeReconstructedFace();
		break;
	}
	case 3:
	{
		auto& renderer = Renderer::Get();
		Face face = Face("sample1", "BFM17");
		// randomize for testing
		optimizer.optimize(face);
		Matrix4f projection_matrix = face.getExtrinsics().cast<float>();
		Matrix4f intrinsics = Matrix4f::Identity();
		intrinsics.block(0, 0, 3, 3) = face.getIntrinsics().cast<float>();
		projection_matrix = intrinsics * projection_matrix;
		/*Face face;
		face.randomizeParameters();*/
		renderer.initialiaze_rendering_context(face.getFaceModel(), 720, 720);

		// Initialize the default projection matrix which moves the face to origin
		double scale_factor = 1 / 100.;
		MatrixXd coords = face.calculateVerticesDefault().reshaped(3, face.getFaceModel().getNumVertices()).transpose();
		Vector3d mean = coords.colwise().mean();
		/*Matrix4d projection_matrix_ = Matrix4d::Identity() * scale_factor;
		projection_matrix_.block(0, 3, 3, 1) = -mean * scale_factor;
		projection_matrix_(3, 3) = 1.;*/
		cout << mean << endl;
		cout << face.getExtrinsics().block(0, 3, 3, 1) << endl;
		
		VectorXf vertices = face.calculateVerticesDefault().cast<float>();
		VectorXf colors = face.calculateColorsDefault().cast<float>();

		renderer.render(face, projection_matrix.transpose().cast<float>(), vertices, colors);

		cv::Mat depth_to_visualize;
		cv::normalize(renderer.get_depth_buffer(), depth_to_visualize, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		cv::imwrite("../../data/samples/2d face image/sample_image_depth.png", depth_to_visualize);
		cv::imwrite("../../data/samples/2d face image/sample_image.png", renderer.get_color_buffer());

		renderer.terminate_rendering_context();
		break;
	}
	default:
		break;
	}
}


int main(int argc, char** argv) {
	omp_set_num_threads(omp_get_max_threads());
	google::InitGoogleLogging(argv[0]);
	handleMenu();
	performTask();
	return 0;
}