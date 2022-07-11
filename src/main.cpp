#include <iostream>
#include "Optimizer.h"
#include "Face.h"
#include "Renderer.h"
#include <chrono>

using namespace std;

vector<string> taskOptions{ "Face reconstruction", "Expression transfer", "Rasterize random face" };
int taskOption = -1;
// For now we only hadle sample images case
// vector<string> inputOptions{ "Use sample image(s)" };
// int inputOption = -1;

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
		Face face;
		face.randomizeParameters();

		// Initialize the default projection matrix which moves the face to origin
		double scale_factor = 1 / 100.;
		MatrixXd coords = face.calculateVerticesDefault().reshaped(3, face.getFaceModel().getNumVertices()).transpose();
		Vector3d mean = coords.colwise().mean();
		Matrix4d projection_matrix = Matrix4d::Identity() * scale_factor;
		projection_matrix.block(0, 3, 3, 1) = -mean * scale_factor;
		projection_matrix(3, 3) = 1.;
		
		const auto start = std::chrono::steady_clock::now();
		auto buffers = render(face, projection_matrix.transpose().cast<float>(), 720, 720);
		const auto end = std::chrono::steady_clock::now();
		std::cout << "time used: " << std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) << "ms\n";

		cv::Mat depth_to_visualize;
		cv::normalize(std::get<1>(buffers), depth_to_visualize, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		cv::imwrite("../../data/samples/2d face image/sample_image_depth.png", depth_to_visualize);
		cv::imwrite("../../data/samples/2d face image/sample_image.png", std::get<0>(buffers));
		break;
	}
	default:
		break;
	}
}


int main(int argc, char** argv) {
	google::InitGoogleLogging(argv[0]);
	handleMenu();
	performTask();
	return 0;
}