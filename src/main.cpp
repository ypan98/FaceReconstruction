#include<iostream>
#include "Optimizer.h"
#include "Face.h"
#include "Renderer.h"
#include <chrono>

using namespace std;

// singleton class
Renderer Renderer::s_instance;

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
		sourceFace.randomizeParameters(1,1,1);
		optimizer.optimize(sourceFace);
		// write out mesh
		sourceFace.writeReconstructedFace();
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
		float scale_factor = 1 / 100.f;
		Face face;
		face.randomizeParameters();
		MatrixXf coords = face.calculateVerticesDefault().reshaped(3, face.getFaceModel().getNumVertices()).transpose();
		Vector3f mean = coords.colwise().mean();
		Matrix4f projection_matrix = Matrix4f::Identity() * scale_factor;
		projection_matrix.block(0, 3, 3, 1) = -mean * scale_factor;
		projection_matrix(3, 3) = 1.f;
		const auto start = std::chrono::steady_clock::now();
		cv::Mat img = renderer.render(face, projection_matrix, 720, 720);
		const auto end = std::chrono::steady_clock::now();
		std::cout << "time used: " << std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) << "ms\n";
		cv::imwrite("../../data/samples/2d face image/sample_image.png", img);
		break;
	}
	default:
		break;
	}
}


int main() {
	handleMenu();
	performTask();
	return 0;
}