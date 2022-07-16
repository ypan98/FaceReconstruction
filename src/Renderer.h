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

	void initialiaze_rendering_context(FaceModel& face_model, int height, int width);

	void terminate_rendering_context();

	void clear_buffers();

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

	cv::Mat get_pixel_normal_buffer() {
		return pixel_normal_buffer;
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

	void render(Face& face, Matrix4f& projectionMatrix, VectorXf& vertices, VectorXf& colors, VectorXf& sh_coefficients);

private:
	static Renderer s_instance;
	int viewport_height, viewport_width;
	cudaStream_t streams[9];
	float* device_vertices, * device_colors, * device_projection, * device_sh_coefficients;
	double* device_bary_centric, * device_point_normals;
	unsigned char* device_rendered_color, *device_should_render_fragment;
	int* device_pixel_triangle, * device_triangles, * device_depth_locked, * device_depth;
	cv::Mat color_img, depth_img, pixel_bary_coord_buffer, pixel_triangle_buffer, pixel_normal_buffer;
};