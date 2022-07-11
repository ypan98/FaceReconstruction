#pragma once
#include <iostream>
#include <opencv2/core.hpp>
#include <Eigen.h>
#include "Face.h"

std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> render(Face& face, Matrix4d projectionMatrix, int height, int width);