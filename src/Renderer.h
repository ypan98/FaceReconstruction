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

class Renderer {
public:
	static Renderer& Get() {
		return s_instance;
	}

	void initialiaze_rendering_context(FaceModel& face_model, int height, int width) {
		viewport_height = height;
		viewport_width = width;

		color_img = cv::Mat::zeros(viewport_height, viewport_width, CV_8UC3);
		depth_img = INT_MAX * cv::Mat::ones(viewport_height, viewport_width, CV_32SC1);
		pixel_bary_coord_buffer = cv::Mat::zeros(viewport_height, viewport_width, CV_64FC3);
		pixel_triangle_buffer = cv::Mat::zeros(viewport_height, viewport_width, CV_32S);

		MatrixXi triangles = face_model.getTriangulation().transpose();
		MatrixXf shape_var = face_model.getShapeBasisStdMultiplied().cast<float>();
		MatrixXf color_var = face_model.getColoBasisStdMultiplied().cast<float>();
		MatrixXf exp_var = face_model.getExpBasisStdMultiplied().cast<float>();

		VectorXf shape_mean = face_model.getShapeMean().cast<float>();
		VectorXf color_mean = face_model.getColorMean().cast<float>();
		VectorXf exp_mean = face_model.getExpMean().cast<float>();

		int num_vertices = face_model.getNumVertices();
		int num_triangles = triangles.cols();

		cudaStream_t streams[15];
		for (int i = 0; i < 15; ++i) {
			cudaStreamCreate(&streams[i]);
		}
		
		// Allocate memory for buffers
		cudaMallocAsync((void**)&device_vertices, num_vertices * 3 * sizeof(float), streams[0]);
		cudaMallocAsync((void**)&device_colors, num_vertices * 3 * sizeof(float), streams[1]);
		cudaMallocAsync((void**)&device_triangles, num_triangles * 3 * sizeof(int), streams[2]);
		cudaMallocAsync((void**)&device_rendered_color, viewport_height * viewport_width * 3 * sizeof(unsigned char), streams[3]);
		cudaMallocAsync((void**)&device_depth, viewport_height * viewport_width * sizeof(int), streams[4]);
		cudaMallocAsync((void**)&device_bary_centric, viewport_height * viewport_width * 3 * sizeof(double), streams[5]);
		cudaMallocAsync((void**)&device_pixel_triangle, viewport_height * viewport_width * sizeof(int), streams[6]);
		cudaMallocAsync((void**)&device_projection, 16 * sizeof(float), streams[7]);
		cudaMallocAsync((void**)&device_depth_locked, viewport_height * viewport_width * sizeof(int), streams[8]);

		// Allocate memory for face model parameters
		cudaMallocAsync((void**)&device_model_shape_var, shape_var.rows() * shape_var.cols() * sizeof(float), streams[9]);
		cudaMallocAsync((void**)&device_model_exp_var, exp_var.rows() * exp_var.cols() * sizeof(float), streams[10]);
		cudaMallocAsync((void**)&device_model_color_var, color_var.rows() * color_var.cols() * sizeof(float), streams[11]);
		cudaMallocAsync((void**)&device_model_shape_mean, num_vertices * 3 * sizeof(float), streams[12]);
		cudaMallocAsync((void**)&device_model_exp_mean, num_vertices * 3 * sizeof(float), streams[13]);
		cudaMallocAsync((void**)&device_model_color_mean, num_vertices * 3 * sizeof(float), streams[14]);

		cudaDeviceSynchronize();

		// Initialiaze buffers
		cudaMemcpyAsync(device_triangles, triangles.data(), num_triangles * 3 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(device_depth, depth_img.data, viewport_height * viewport_width * sizeof(int), cudaMemcpyHostToDevice);

		cudaMemcpyAsync(device_model_shape_var, shape_var.data(), shape_var.rows() * shape_var.cols() * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(device_model_exp_var, exp_var.data(), exp_var.rows() * exp_var.cols() * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(device_model_color_var, color_var.data(), color_var.rows() * color_var.cols() * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(device_model_shape_mean, shape_mean.data(), num_vertices * 3 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(device_model_exp_mean, exp_mean.data(), num_vertices * 3 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(device_model_color_mean, color_mean.data(), num_vertices * 3 * sizeof(float), cudaMemcpyHostToDevice);

		cudaMemsetAsync(device_rendered_color, 0, viewport_height * viewport_width * 3 * sizeof(unsigned char), streams[0]);
		cudaMemsetAsync(device_bary_centric, 0, viewport_height * viewport_width * 3 * sizeof(double), streams[1]);
		cudaMemsetAsync(device_pixel_triangle, 0, viewport_height * viewport_width * sizeof(int), streams[2]);
		cudaMemsetAsync(device_depth_locked, 0, viewport_height * viewport_width * sizeof(int), streams[3]);

		cudaDeviceSynchronize();
	}

	void terminate_rendering_context() {
		cudaFreeAsync(device_vertices, streams[0]);
		cudaFreeAsync(device_colors, streams[1]);
		cudaFreeAsync(device_projection, streams[2]);
		cudaFreeAsync(device_rendered_color, streams[3]);
		cudaFreeAsync(device_depth, streams[4]);
		cudaFreeAsync(device_bary_centric, streams[5]);
		cudaFreeAsync(device_pixel_triangle, streams[6]);
		cudaFreeAsync(device_triangles, streams[7]);
		cudaFreeAsync(device_depth_locked, streams[8]);
		cudaFreeAsync(device_model_shape_var, streams[9]);
		cudaFreeAsync(device_model_color_var, streams[10]);
		cudaFreeAsync(device_model_exp_var, streams[11]);
		cudaFreeAsync(device_model_shape_mean, streams[12]);
		cudaFreeAsync(device_model_color_mean, streams[13]);
		cudaFreeAsync(device_model_exp_mean, streams[14]);

		for (int i = 0; i < 15; ++i) {
			cudaStreamDestroy(streams[i]);
		}
	}

	void clear_buffers() {
		cv::Mat depth = INT_MAX * cv::Mat::ones(viewport_height, viewport_width, CV_32SC1);
		cudaMemcpyAsync(device_depth, depth.data, viewport_height * viewport_width * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemsetAsync(device_rendered_color, 0, viewport_height * viewport_width * 3 * sizeof(unsigned char), streams[0]);
		cudaMemsetAsync(device_bary_centric, 0, viewport_height * viewport_width * 3 * sizeof(double), streams[1]);
		cudaMemsetAsync(device_pixel_triangle, 0, viewport_height * viewport_width * sizeof(int), streams[2]);
		cudaMemsetAsync(device_depth_locked, 0, viewport_height * viewport_width * sizeof(int), streams[3]);
	}

	cv::Mat get_color_buffer() {
		return color_img;
	}

	cv::Mat get_depth_buffer() {
		return depth_img;
	}

	cv::Mat get_pixel_bary_coord_buffer() {
		return pixel_bary_coord_buffer;
	}

	cv::Mat get_pixel_triangle_buffer() {
		return pixel_triangle_buffer;
	}

	void render(Face& face, Matrix4f projectionMatrix);

private:
	static Renderer s_instance;
	int viewport_height, viewport_width;
	cudaStream_t streams[15];
	float* device_vertices, * device_colors, * device_projection;
	float* device_model_shape_var, * device_model_exp_var, * device_model_color_var;
	float* device_model_shape_mean, * device_model_exp_mean, * device_model_color_mean;
	double* device_bary_centric;
	unsigned char* device_rendered_color;
	int* device_pixel_triangle, * device_triangles, * device_depth_locked, * device_depth;
	cv::Mat color_img, depth_img, pixel_bary_coord_buffer, pixel_triangle_buffer;
};