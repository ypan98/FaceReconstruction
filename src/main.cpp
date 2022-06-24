#include<iostream>
using namespace std;

int imgOption = -1;
int taskOption = -1;

void handleMenu() {
	cout << "Please select an option for the input image(s):\n" << "1. Use sample image(s)\n" << "2. Take photos\n" << "3. Real-time streaming\n";
	while (cin >> imgOption) {
		if (imgOption >= 0 && imgOption <= 3) break;
		else cout << "Enter a valid option\n";
	}
	cout << "Please select an option for the task:\n" << "1. Face reconstruction\n" << "2. Expression transfer\n";
	while (cin >> taskOption) {
		if (taskOption == 1 || taskOption == 2) break;
		else cout << "Enter a valid option\n";
	}
}

void main() {
	handleMenu();
}