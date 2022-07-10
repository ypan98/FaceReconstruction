#include <cmath>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda.h>
#include "Renderer.h"


struct __align__(16) ModelVertex
{
	float4 position;
	float4 color;
};

// Type definitions
struct __align__(16) TriangleToRasterize
{
	ModelVertex v0, v1, v2;

	int min_x;
	int max_x;
	int min_y;
	int max_y;

	double one_over_z0;
	double one_over_z1;
	double one_over_z2;
};

// Auxiliary functions
__device__ inline float4 projection(float* projection_matrix, float4& vertex) {
	float projected_x = projection_matrix[0] * vertex.x + projection_matrix[1] * vertex.y + projection_matrix[2] * vertex.z + projection_matrix[3] * vertex.w;
	float projected_y = projection_matrix[4] * vertex.x + projection_matrix[5] * vertex.y + projection_matrix[6] * vertex.z + projection_matrix[7] * vertex.w;
	float projected_z = projection_matrix[8] * vertex.x + projection_matrix[9] * vertex.y + projection_matrix[10] * vertex.z + projection_matrix[11] * vertex.w;
	float projected_w = projection_matrix[12] * vertex.x + projection_matrix[13] * vertex.y + projection_matrix[14] * vertex.z + projection_matrix[15] * vertex.w;
	return make_float4(projected_x, projected_y, projected_z, projected_w);
}

// Main functions
__device__ bool vertices_ccw_in_screen_space(const float4& v0, const float4& v1, const float4& v2)
{
	float dx01 = v1.x - v0.x;
	float dy01 = v1.y - v0.y;
	float dx02 = v2.x - v0.x;
	float dy02 = v2.y - v0.y;

	return (dx01 * dy02 - dy01 * dx02 < 0.0f);
}

__device__ float2 clip_to_screen_space(const float2& clip_coords, int screen_width, int screen_height)
{
	const float x_ss = (clip_coords.x + 1.0f) * (screen_width / 2.0f);
	const float y_ss = screen_height - (clip_coords.y + 1.0f) * (screen_height / 2.0f);
	return make_float2(x_ss, y_ss);
}

__device__ float4 calculate_clipped_bounding_box(float4& v0, float4& v1, float4& v2, int viewport_width, int viewport_height)
{
	int minX = fmaxf(fminf(floorf(v0.x), fminf(floorf(v1.x), floorf(v2.x))), 0.0f);
	int maxX = fminf(fmaxf(ceilf(v0.x), fmaxf(ceilf(v1.x), ceilf(v2.x))), static_cast<float>(viewport_width - 1));
	int minY = fmaxf(fminf(floorf(v0.y), fminf(floorf(v1.y), floorf(v2.y))), 0.0f);
	int maxY = fminf(fmaxf(ceilf(v0.y), fmaxf(ceilf(v1.y), ceilf(v2.y))), static_cast<float>(viewport_height - 1));
	return make_float4(minX, minY, maxX - minX, maxY - minY);
};

__device__ TriangleToRasterize process_prospective_tri(ModelVertex v0, ModelVertex v1, ModelVertex v2, int viewport_width, int viewport_height, bool enable_backface_culling,
	bool& should_render)
{
	TriangleToRasterize t;
	t.v0 = v0;
	t.v1 = v1;
	t.v2 = v2;

	// Only for texturing or perspective texturing:
	t.one_over_z0 = 1.0 / (double)t.v0.position.w;
	t.one_over_z1 = 1.0 / (double)t.v1.position.w;
	t.one_over_z2 = 1.0 / (double)t.v2.position.w;

	// divide by w
	t.v0.position = make_float4(t.v0.position.x / t.v0.position.w, t.v0.position.y / t.v0.position.w, t.v0.position.z / t.v0.position.w, t.v0.position.w / t.v0.position.w);
	t.v1.position = make_float4(t.v1.position.x / t.v1.position.w, t.v1.position.y / t.v1.position.w, t.v1.position.z / t.v1.position.w, t.v1.position.w / t.v1.position.w);
	t.v2.position = make_float4(t.v2.position.x / t.v2.position.w, t.v2.position.y / t.v2.position.w, t.v2.position.z / t.v2.position.w, t.v2.position.w / t.v2.position.w);

	float2 v0_screen = clip_to_screen_space(make_float2(t.v0.position.x, t.v0.position.y), viewport_width, viewport_height);
	t.v0.position.x = v0_screen.x;
	t.v0.position.y = v0_screen.y;
	float2 v1_screen = clip_to_screen_space(make_float2(t.v1.position.x, t.v1.position.y), viewport_width, viewport_height);
	t.v1.position.x = v1_screen.x;
	t.v1.position.y = v1_screen.y;
	float2 v2_screen = clip_to_screen_space(make_float2(t.v2.position.x, t.v2.position.y), viewport_width, viewport_height);
	t.v2.position.x = v2_screen.x;
	t.v2.position.y = v2_screen.y;

	if (enable_backface_culling) {
		if (!vertices_ccw_in_screen_space(t.v0.position, t.v1.position, t.v2.position)) {
			should_render = false;
			return t;
		}
	}

	// Get the bounding box of the triangle:
	float4 boundingBox = calculate_clipped_bounding_box(t.v0.position, t.v1.position, t.v2.position, viewport_width, viewport_height);
	t.min_x = boundingBox.x;
	t.max_x = boundingBox.x + boundingBox.z;
	t.min_y = boundingBox.y;
	t.max_y = boundingBox.y + boundingBox.w;

	if (t.max_x <= t.min_x || t.max_y <= t.min_y) {
		should_render = false;
		return t;
	}

	should_render = true;
	return t;
};

__device__ double implicit_line(float x, float y, const float4& v1, const float4& v2)
{
	return ((double)v1.y - (double)v2.y) * (double)x + ((double)v2.x - (double)v1.x) * (double)y + (double)v1.x * (double)v2.y - (double)v2.x * (double)v1.y;
};

__device__ void raster_triangle(TriangleToRasterize triangle, unsigned char* color_buffer, int* depth_buffer, double* pixel_bary_coord_buffer, 
	int* pixel_triangle_buffer, int* depth_locked, int triangle_index, int width)
{
	for (int yi = triangle.min_y; yi <= triangle.max_y; ++yi)
	{
		for (int xi = triangle.min_x; xi <= triangle.max_x; ++xi)
		{
			const float x = static_cast<float>(xi) + 0.5f;
			const float y = static_cast<float>(yi) + 0.5f;

			// these will be used for barycentric weights computation
			const double one_over_v0ToLine12 = 1.0 / implicit_line(triangle.v0.position.x, triangle.v0.position.y, triangle.v1.position, triangle.v2.position);
			const double one_over_v1ToLine20 = 1.0 / implicit_line(triangle.v1.position.x, triangle.v1.position.y, triangle.v2.position, triangle.v0.position);
			const double one_over_v2ToLine01 = 1.0 / implicit_line(triangle.v2.position.x, triangle.v2.position.y, triangle.v0.position, triangle.v1.position);
			// affine barycentric weights
			double alpha = implicit_line(x, y, triangle.v1.position, triangle.v2.position) * one_over_v0ToLine12;
			double beta = implicit_line(x, y, triangle.v2.position, triangle.v0.position) * one_over_v1ToLine20;
			double gamma = implicit_line(x, y, triangle.v0.position, triangle.v1.position) * one_over_v2ToLine01;

			// if pixel (x, y) is inside the triangle or on one of its edges
			if (alpha >= 0 && beta >= 0 && gamma >= 0)
			{
				const int pixel_index_row = yi;
				const int pixel_index_col = xi;

				double z_affine = alpha * static_cast<double>(triangle.v0.position.z) + beta * static_cast<double>(triangle.v1.position.z) + gamma * static_cast<double>(triangle.v2.position.z);

				if (z_affine > 1.0)
				{
					continue;
				}
				int index = pixel_index_row * width + pixel_index_col;

				bool isLocked = false;
				do
				{
					isLocked = (atomicCAS(&depth_locked[index], 0, 1) == 0);
					int depth = z_affine * INT_MAX;
					atomicMin(&depth_buffer[index], depth);
					if (depth_buffer[index] == depth) {
						// perspective-correct barycentric weights
						double d = alpha * triangle.one_over_z0 + beta * triangle.one_over_z1 + gamma * triangle.one_over_z2;
						d = 1.0 / d;
						alpha *= d * triangle.one_over_z0;
						beta *= d * triangle.one_over_z1;
						gamma *= d * triangle.one_over_z2;

						// attributes interpolation
						double red_ = alpha * static_cast<double>(triangle.v0.color.x) + beta * static_cast<double>(triangle.v1.color.x) + gamma * static_cast<double>(triangle.v2.color.x);
						double blue_ = alpha * static_cast<double>(triangle.v0.color.y) + beta * static_cast<double>(triangle.v1.color.y) + gamma * static_cast<double>(triangle.v2.color.y);
						double green_ = alpha * static_cast<double>(triangle.v0.color.z) + beta * static_cast<double>(triangle.v1.color.z) + gamma * static_cast<double>(triangle.v2.color.z);

						// clamp bytes to 255
						const unsigned char red = static_cast<unsigned char>(255.0f * fminf(static_cast<float>(red_), 1.0f));
						const unsigned char green = static_cast<unsigned char>(255.0f * fminf(static_cast<float>(blue_), 1.0f));
						const unsigned char blue = static_cast<unsigned char>(255.0f * fminf(static_cast<float>(green_), 1.0f));

						// update buffers
						color_buffer[index * 3] = blue;
						color_buffer[index * 3 + 1] = green;
						color_buffer[index * 3 + 2] = red;

						pixel_bary_coord_buffer[index * 3] = alpha;
						pixel_bary_coord_buffer[index * 3 + 1] = gamma;
						pixel_bary_coord_buffer[index * 3 + 2] = beta;

						pixel_triangle_buffer[index] = triangle_index;
					}

					if (isLocked) {
						depth_locked[index] = 0;
					}
				} while (!isLocked);
			}
		}
	}
};

__global__ void render_(int* indices_buffer, float* vertex_position_buffer, float* vertex_color_buffer,
	unsigned char* color_buffer, int* depth_buffer, double* pixel_bary_coord_buffer, int* pixel_triangle_buffer,
	float* projection_matrix, int* depth_locked,
	int max_triangle_id,
	int viewport_width, int viewport_height)
{
	int triangle_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (triangle_index < max_triangle_id) {
		ModelVertex vertices[3];
		for (int i = 0; i < 3; ++i) {
			int v_id = indices_buffer[3 * triangle_index + i];
			float4 vertex_position = make_float4(vertex_position_buffer[v_id * 3], vertex_position_buffer[v_id * 3 + 1], vertex_position_buffer[v_id * 3 + 2], 1.f);
			vertices[i].position = projection(projection_matrix, vertex_position);
		}

		TriangleToRasterize triangles_to_raster;
		unsigned char visibility_bits[3];
		for (unsigned char k = 0; k < 3; k++)
		{
			visibility_bits[k] = 0;

			float x_cc = vertices[k].position.x;
			float y_cc = vertices[k].position.y;
			float z_cc = vertices[k].position.z;
			float w_cc = vertices[k].position.w;

			if (x_cc < -w_cc)
				visibility_bits[k] |= 1;
			if (x_cc > w_cc)
				visibility_bits[k] |= 2;
			if (y_cc < -w_cc)
				visibility_bits[k] |= 4;
			if (y_cc > w_cc)
				visibility_bits[k] |= 8;
			if (z_cc < -w_cc)
				visibility_bits[k] |= 16;
			if (z_cc > w_cc)
				visibility_bits[k] |= 32;
		} // if all bits are 0, then it's inside the frustum

		// all vertices are not visible - reject the triangle.
		if ((visibility_bits[0] & visibility_bits[1] & visibility_bits[2]) > 0)
		{
			return;
		}
		bool should_render = false;
		triangles_to_raster = process_prospective_tri(vertices[0], vertices[1], vertices[2], viewport_width, viewport_height, true, should_render);
		if (should_render) {
			raster_triangle(triangles_to_raster, color_buffer, depth_buffer, pixel_bary_coord_buffer, pixel_triangle_buffer, depth_locked, triangle_index, viewport_width);
		}
	}
}

std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> render(Face & face, Matrix4f projectionMatrix, int height, int width) {
	VectorXf vertices = face.calculateVerticesDefault();
	VectorXf colors = face.calculateColorsDefault();
	MatrixXi triangles = face.getFaceModel().getTriangulation().transpose();
	int num_vertices = face.getFaceModel().getNumVertices();
	int num_triangles = triangles.cols();

	cv::Mat color_img = cv::Mat::zeros(height, width, CV_8UC3);
	cv::Mat depth_img = INT_MAX * cv::Mat::ones(height, width, CV_32SC1);
	cv::Mat pixel_bary_coord_buffer = cv::Mat::zeros(height, width, CV_64FC3);
	cv::Mat pixel_triangle_buffer = cv::Mat::zeros(height, width, CV_32S);

	float* device_vertices, * device_colors, * device_projection;
	unsigned char* device_rendered_color;
	double* device_bary_centric;
	int* device_pixel_triangle, * device_triangles, * device_depth_locked, * device_depth;

	cudaStream_t streams[9];

	for (int i = 0; i < 9; ++i) {
		cudaStreamCreate(&streams[i]);
	}
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	cudaMallocAsync((void**)&device_vertices, num_vertices * 3 * sizeof(float), streams[0]);
	cudaMallocAsync((void**)&device_colors, num_vertices * 3 * sizeof(float), streams[1]);
	cudaMallocAsync((void**)&device_triangles, num_triangles * 3 * sizeof(int), streams[2]);
	cudaMallocAsync((void**)&device_rendered_color, height * width * 3 * sizeof(unsigned char), streams[3]);
	cudaMallocAsync((void**)&device_depth, height * width * sizeof(int), streams[4]);
	cudaMallocAsync((void**)&device_bary_centric, height * width * 3 * sizeof(double), streams[5]);
	cudaMallocAsync((void**)&device_pixel_triangle, height * width * sizeof(int), streams[6]);
	cudaMallocAsync((void**)&device_projection, 16 * sizeof(float), streams[7]);
	cudaMallocAsync((void**)&device_depth_locked, height * width * sizeof(int), streams[8]);

	cudaDeviceSynchronize();

	cudaMemcpyAsync(device_projection, projectionMatrix.data(), 16 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(device_vertices, vertices.data(), num_vertices * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(device_colors, colors.data(), num_vertices * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(device_triangles, triangles.data(), num_triangles * 3 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(device_depth, depth_img.data, height * width * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemsetAsync(device_rendered_color, 0, height * width * 3 * sizeof(unsigned char), streams[0]);
	cudaMemsetAsync(device_bary_centric, 0, height * width * 3 * sizeof(double), streams[1]);
	cudaMemsetAsync(device_pixel_triangle, 0, height * width * sizeof(int), streams[2]);
	cudaMemsetAsync(device_depth_locked, 0, height * width * sizeof(int), streams[3]);

	cudaDeviceSynchronize();

	int block_size = 256;
	int grid_size = int(num_triangles / block_size) + 1;

	render_<<<grid_size, block_size>>> (device_triangles, device_vertices, device_colors, device_rendered_color,
		device_depth, device_bary_centric, device_pixel_triangle, device_projection, device_depth_locked, num_triangles, width, height);

	cudaDeviceSynchronize();

	cudaMemcpyAsync(color_img.data, device_rendered_color, height * width * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(depth_img.data, device_depth, height * width * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(pixel_bary_coord_buffer.data, device_bary_centric, height * width * 3 * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(pixel_triangle_buffer.data, device_pixel_triangle, height * width * sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cudaFreeAsync(device_vertices, streams[0]);
	cudaFreeAsync(device_colors, streams[1]);
	cudaFreeAsync(device_projection, streams[2]);
	cudaFreeAsync(device_rendered_color, streams[3]);
	cudaFreeAsync(device_depth, streams[4]);
	cudaFreeAsync(device_bary_centric, streams[5]);
	cudaFreeAsync(device_pixel_triangle, streams[6]);
	cudaFreeAsync(device_triangles, streams[7]);

	for (int i = 0; i < 8; ++i) {
		cudaStreamDestroy(streams[i]);
	}
	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::string time = std::to_string(milliseconds);
	std::printf(time.c_str());

	return std::tuple(color_img, depth_img, pixel_bary_coord_buffer, pixel_triangle_buffer);
}