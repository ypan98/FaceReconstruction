#include<iostream>
#include "Optimizer.h"
#include "Face.h"

using namespace std;

vector<string> taskOptions{ "Face reconstruction", "Expression transfer" };
int taskOption = -1;
// For now we only hadle sample images case
// vector<string> inputOptions{ "Use sample image(s)" };
// int inputOption = -1;


void handleMenu() {
	cout << "Please select a task:\n";
	for (int i = 0; i < taskOptions.size(); i++) cout << i+1 << ". " << taskOptions[i] << endl;
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
		optimizer.optimize(sourceFace);
		// write out mesh
		sourceFace.randomizeParameters();
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
	default:
		break;
	}
}


int main() {
	handleMenu();
	performTask();
	return 0;
}