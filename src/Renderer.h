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
#include <stdlib.h>
#include <cuda/std/cmath>

class Renderer {
public:
	static Renderer& Get() {
		return s_instance;
	}

	void initialiaze_rendering_context(FaceModel& face_model, int height, int width);

	void terminate_rendering_context();

	void clear_buffers();

	cv::Mat get_color_buffer() {
		return color_img;
	}

	cv::Mat get_depth_buffer() {
		cv::Mat depth_to_visualize;
		cv::normalize(depth_img, depth_to_visualize, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		return depth_to_visualize;
	}

	cv::Mat get_pixel_bary_coord_buffer() {
		return pixel_bary_coord_buffer;
	}

	cv::Mat get_pixel_triangle_buffer() {
		return pixel_triangle_buffer;
	}

	cv::Mat get_pixel_triangle_normal_buffer() {
		return pixel_triangle_normals_buffer;
	}

	Matrix4d get_perspective_projection_matrix(double fov, double aspect_ratio, double z_near=0.1, double z_far=100000.) {
		Matrix4d perspective_projection_matrix = Matrix4d::Zero();
		perspective_projection_matrix(0, 0) = fov / aspect_ratio;
		perspective_projection_matrix(1, 1) = fov;
		perspective_projection_matrix(2, 2) = (z_near + z_far) / (z_near - z_far);
		perspective_projection_matrix(2, 3) = (2 * z_near * z_far) / (z_near - z_far);
		perspective_projection_matrix(3, 2) = -1;
		return perspective_projection_matrix;
	}

	void render(Matrix4f& mvp_matrix, Matrix4f& mv_matrix, VectorXf& vertices, VectorXf& colors, VectorXf& sh_red_coefficients,
		VectorXf& sh_green_coefficients, VectorXf& sh_blue_coefficients);

private:
	static Renderer s_instance;
	// Face model assigned to this render
	int viewport_height, viewport_width, num_vertices, num_triangles;
	
	// Cuda streams used to parallelize GPU IO operations
	cudaStream_t streams[13];
	
	// Device buffers used to store face parameters
	float* device_vertices, * device_vertex_normals, * device_colors;
	int* device_triangles;

	// Device buffers used to store camera and illumination settings
	float* device_mvp, * device_mv, * device_sh_coefficients;

	// Device buffers used to store the rendered image
	// and other information which are required to
	// compute the analytical partial derivatives with respect to P
	unsigned char* device_rendered_color;
	double* device_pixel_bary_coord;
	int* device_pixel_triangle, * device_depth;
	float* device_pixel_triangle_normals;

	// Device buffer used to control sequential IO cuda operations
	int* device_depth_locked;

	// Cpu buffers used to receive information returned by render
	cv::Mat color_img, depth_img, pixel_bary_coord_buffer, pixel_triangle_buffer, pixel_triangle_normals_buffer;
};