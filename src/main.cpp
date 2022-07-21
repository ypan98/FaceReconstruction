#include <iostream>
#include "Optimizer.h"
#include "Face.h"
#include "Renderer.h"
#include <chrono>
#include <omp.h>

using namespace std;

vector<string> taskOptions{ "Face reconstruction", "Expression transfer"};
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
		auto render = Renderer::Get();
		Face sourceFace = Face("sample2", "BFM17");
		render.initialiaze_rendering_context(sourceFace.getFaceModel(), sourceFace.getImage().getHeight(), sourceFace.getImage().getWidth());
		sourceFace.setIntrinsics(double(60), double(sourceFace.getImage().getWidth()) / double(sourceFace.getImage().getHeight()),
			double(11000), double(12000));
		optimizer.optimize(sourceFace, 0);
		Matrix4f mvp_matrix = sourceFace.getFullProjectionMatrix().transpose().cast<float>();
		Matrix4f mv_matrix = sourceFace.getExtrinsics().transpose().cast<float>();
		VectorXf vertices = sourceFace.getShape().cast<float>();
		VectorXf colors = sourceFace.getColor().cast<float>();
		VectorXf sh_red_coefficients = sourceFace.getSHRedCoefficients().cast<float>();
		VectorXf sh_green_coefficients = sourceFace.getSHGreenCoefficients().cast<float>();
		VectorXf sh_blue_coefficients = sourceFace.getSHBlueCoefficients().cast<float>();
		render.render(mvp_matrix, mv_matrix, vertices, colors, sh_red_coefficients, sh_green_coefficients, sh_blue_coefficients, sourceFace.get_z_near(),
			sourceFace.get_z_far());
		sourceFace.setColor(render.get_re_rendered_vertex_color().cast<double>());
		imshow("face", render.get_color_buffer());
		cv::waitKey(0);
		cout << sourceFace.getSHRedCoefficients() << endl;
		cout << sourceFace.getSHGreenCoefficients() << endl;
		cout << sourceFace.getSHBlueCoefficients() << endl;

		// write out mesh
		sourceFace.writeReconstructedFace();
		break;
	}
	case 2:
	{
		// reconstruct source face
		Face sourceFace = Face("sample1", "BFM17");
		optimizer.optimize(sourceFace, 0);
		// reconstruct target face
		Face targetFace = Face("sample2", "BFM17");
		optimizer.optimize(targetFace, 0);
		// transfer expression and write out mesh
		targetFace.transferExpression(sourceFace);
		targetFace.writeReconstructedFace();
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