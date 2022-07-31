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
	Renderer() {}

	Renderer(FaceModel& face_model, int height, int width) {
		initialiaze_rendering_context(face_model, height, width);
	}

	void initialiaze_rendering_context(FaceModel& face_model, int height, int width);

	void terminate_rendering_context();

	void clear_buffers();

	void render(Matrix4f& mvp_matrix, Matrix4f& mv_matrix, VectorXf& vertices, VectorXf& colors, VectorXf& sh_red_coefficients,
		VectorXf& sh_green_coefficients, VectorXf& sh_blue_coefficients, float z_near, float z_far);

	// getters and setters
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

	cv::Mat get_pixel_triangle_normal_buffer() {
		return pixel_triangle_normals_buffer;
	}

	VectorXf get_re_rendered_vertex_color() {
		return re_rendered_vertex_color;
	}

private:
	// Face model assigned to this render
	int viewport_height, viewport_width, num_vertices, num_triangles;
	
	// Cuda streams used to parallelize GPU IO operations
	cudaStream_t streams[14];
	
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
	float* device_pixel_triangle_normals, * device_depth_to_visualize;

	// Device buffer used to control sequential IO cuda operations
	int* device_depth_locked;

	// Cpu buffers used to receive information returned by render
	cv::Mat color_img, depth_img, pixel_bary_coord_buffer, pixel_triangle_buffer, pixel_triangle_normals_buffer;
	VectorXf re_rendered_vertex_color;
};