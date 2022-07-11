#pragma once
#include <iostream>
#include <opencv2/core.hpp>
#include <Eigen.h>
#include "Face.h"
#include <cmath>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda.h>

std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> render(Face& face, Matrix4f projectionMatrix, int height, int width);