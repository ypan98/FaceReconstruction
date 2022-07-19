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

	unsigned char should_draw;
};

// Auxiliary functions
__device__ inline float4 projection(float* projection_matrix, float4& vertex) {
	float projected_x = projection_matrix[0] * vertex.x + projection_matrix[1] * vertex.y + projection_matrix[2] * vertex.z + projection_matrix[3] * vertex.w;
	float projected_y = projection_matrix[4] * vertex.x + projection_matrix[5] * vertex.y + projection_matrix[6] * vertex.z + projection_matrix[7] * vertex.w;
	float projected_z = projection_matrix[8] * vertex.x + projection_matrix[9] * vertex.y + projection_matrix[10] * vertex.z + projection_matrix[11] * vertex.w;
	float projected_w = projection_matrix[12] * vertex.x + projection_matrix[13] * vertex.y + projection_matrix[14] * vertex.z + projection_matrix[15] * vertex.w;
	return make_float4(projected_x, projected_y, projected_z, projected_w);
};

// Main functions
__device__ bool vertices_ccw_in_screen_space(const float4& v0, const float4& v1, const float4& v2)
{
	float dx01 = v1.x - v0.x;
	float dy01 = v1.y - v0.y;
	float dx02 = v2.x - v0.x;
	float dy02 = v2.y - v0.y;

	return (dx01 * dy02 - dy01 * dx02 < 0.0f);
};

__device__ float2 clip_to_screen_space(const float2& clip_coords, int screen_width, int screen_height)
{
	const float x_ss = (clip_coords.x + 1.0f) * (screen_width / 2.0f);
	const float y_ss = screen_height - (clip_coords.y + 1.0f) * (screen_height / 2.0f);
	return make_float2(x_ss, y_ss);
};

__device__ float4 calculate_clipped_bounding_box(float4& v0, float4& v1, float4& v2, int viewport_width, int viewport_height)
{
	int minX = fmaxf(fminf(floorf(v0.x), fminf(floorf(v1.x), floorf(v2.x))), 0.0f);
	int maxX = fminf(fmaxf(ceilf(v0.x), fmaxf(ceilf(v1.x), ceilf(v2.x))), static_cast<float>(viewport_width - 1));
	int minY = fmaxf(fminf(floorf(v0.y), fminf(floorf(v1.y), floorf(v2.y))), 0.0f);
	int maxY = fminf(fmaxf(ceilf(v0.y), fmaxf(ceilf(v1.y), ceilf(v2.y))), static_cast<float>(viewport_height - 1));
	return make_float4(minX, minY, maxX - minX, maxY - minY);
};

__device__ float SH_basis_function(float3& normal, int basis_index) {
	switch (basis_index)
	{
	case 0:
		return 0.282095f * 3.1415926f;
	case 1:
		return -0.488603f * normal.y * 2.094395f;
	case 2:
		return 0.488603f * normal.z * 2.094395f;
	case 3:
		return -0.488603f * normal.x * 2.094395f;
	case 4:
		return 1.092548f * normal.x * normal.y * 0.785398f;
	case 5:
		return -1.092548f * normal.y * normal.z * 0.785398f;
	case 6:
		return 0.315392f * (3.0f * normal.z * normal.z - 1.0f) * 0.785398f;
	case 7:
		return -1.092548f * normal.x * normal.z * 0.785398f;
	case 8:
		return 0.546274f * (normal.x * normal.x - normal.y * normal.y) * 0.785398f;
	default:
		return 0.;
	}
};

__device__ double implicit_line(float x, float y, const float4& v1, const float4& v2)
{
	return ((double)v1.y - (double)v2.y) * (double)x + ((double)v2.x - (double)v1.x) * (double)y + (double)v1.x * (double)v2.y - (double)v2.x * (double)v1.y;
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

__global__ void compute_vertex_normals(int* indices_buffer, float* model_view_matrix, float* vertex_position_buffer, float* vertex_normal_buffer, int max_triangle_id) {
	int triangle_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (triangle_index < max_triangle_id) {
		int v0_id = indices_buffer[3 * triangle_index] * 3;
		int v1_id = indices_buffer[3 * triangle_index + 1] * 3;
		int v2_id = indices_buffer[3 * triangle_index + 2] * 3;

		float4 v0 = make_float4(vertex_position_buffer[v0_id], vertex_position_buffer[v0_id + 1], vertex_position_buffer[v0_id + 2], 1.f);
		float4 v1 = make_float4(vertex_position_buffer[v1_id], vertex_position_buffer[v1_id + 1], vertex_position_buffer[v1_id + 2], 1.f);
		float4 v2 = make_float4(vertex_position_buffer[v2_id], vertex_position_buffer[v2_id + 1], vertex_position_buffer[v2_id + 2], 1.f);

		float4 v0_view = projection(model_view_matrix, v0);
		float4 v1_view = projection(model_view_matrix, v1);
		float4 v2_view = projection(model_view_matrix, v2);

		float3 v_1_0 = make_float3(v1_view.x - v0_view.x, v1_view.y - v0_view.y, v1_view.z - v0_view.z);
		float3 v_2_0 = make_float3(v2_view.x - v0_view.x, v2_view.y - v0_view.y, v2_view.z - v0_view.z);
		float3 normal = make_float3(v_1_0.y * v_2_0.z - v_1_0.z * v_2_0.y, -(v_1_0.x * v_2_0.z - v_1_0.z * v_2_0.x), (v_1_0.x * v_2_0.y - v_1_0.y * v_2_0.x));

		atomicAdd(&vertex_normal_buffer[v0_id], normal.x);
		atomicAdd(&vertex_normal_buffer[v0_id + 1], normal.y);
		atomicAdd(&vertex_normal_buffer[v0_id + 2], normal.z);

		atomicAdd(&vertex_normal_buffer[v1_id], normal.x);
		atomicAdd(&vertex_normal_buffer[v1_id + 1], normal.y);
		atomicAdd(&vertex_normal_buffer[v1_id + 2], normal.z);

		atomicAdd(&vertex_normal_buffer[v2_id], normal.x);
		atomicAdd(&vertex_normal_buffer[v2_id + 1], normal.y);
		atomicAdd(&vertex_normal_buffer[v2_id + 2], normal.z);
	}

}

__global__ void compute_vertex_colors(float* vertex_color_buffer, float* vertex_normal_buffer, float* sh_coefficients, int max_vertex_index) {
	int vertex_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (vertex_index < max_vertex_index) {
		float pi = 3.1415926f;
		float3 unnormalized_normal = make_float3(vertex_normal_buffer[3 * vertex_index], vertex_normal_buffer[3 * vertex_index + 1], vertex_normal_buffer[3 * vertex_index + 2]);
		float magnitude = sqrtf(unnormalized_normal.x * unnormalized_normal.x + unnormalized_normal.y * unnormalized_normal.y + unnormalized_normal.z * unnormalized_normal.z);
		float3 normal = make_float3(unnormalized_normal.x / magnitude, unnormalized_normal.y / magnitude, unnormalized_normal.z / magnitude);

		float red_sh, green_sh, blue_sh;
		red_sh = 0.f;
		green_sh = 0.f;
		blue_sh = 0.f;

		for (int i = 0; i < 9; ++i) {
			float sh_kth_basis = SH_basis_function(normal, i);
			red_sh += sh_coefficients[i] * sh_kth_basis;
			green_sh += sh_coefficients[i + 9] * sh_kth_basis;
			blue_sh += sh_coefficients[i + 18] * sh_kth_basis;
		}

		vertex_color_buffer[3 * vertex_index] *= red_sh / pi;
		vertex_color_buffer[3 * vertex_index + 1] *= green_sh / pi;
		vertex_color_buffer[3 * vertex_index + 2] *= blue_sh / pi;

		if (vertex_color_buffer[3 * vertex_index] < 0.) {
			vertex_color_buffer[3 * vertex_index] = 0.;
		}
		if (vertex_color_buffer[3 * vertex_index + 1] < 0.) {
			vertex_color_buffer[3 * vertex_index + 1] = 0.;
		}
		if (vertex_color_buffer[3 * vertex_index + 2] < 0.) {
			vertex_color_buffer[3 * vertex_index + 2] = 0.;
		}
	}
}


__global__ void build_triangles(int* indices_buffer, float* vertex_position_buffer, float* vertex_color_buffer,
	float* projection_matrix, TriangleToRasterize* triangles, int max_triangle_id, int viewport_width, int viewport_height)
{
	int triangle_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (triangle_index < max_triangle_id) {
		ModelVertex vertices[3];
		for (int i = 0; i < 3; ++i) {
			int v_id = indices_buffer[3 * triangle_index + i];
			float4 vertex_position = make_float4(vertex_position_buffer[v_id * 3], vertex_position_buffer[v_id * 3 + 1], vertex_position_buffer[v_id * 3 + 2], 1.f);
			vertices[i].position = projection(projection_matrix, vertex_position);
			vertices[i].color = make_float4(vertex_color_buffer[v_id * 3], vertex_color_buffer[v_id * 3 + 1], vertex_color_buffer[v_id * 3 + 2], 1.f);
		}

		TriangleToRasterize triangle_to_raster;
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
			triangle_to_raster.should_draw = 0;
		}
		bool should_render = false;
		triangle_to_raster = process_prospective_tri(vertices[0], vertices[1], vertices[2], viewport_width, viewport_height, true, should_render);
		if (should_render) {
			triangle_to_raster.should_draw = 1;
		}
		else triangle_to_raster.should_draw = 0;
		triangles[triangle_index] = triangle_to_raster;
	}
}

__global__ void raster_triangle(TriangleToRasterize* triangles, unsigned char* color_buffer, float* vertex_normal_buffer, int* depth_buffer, double* pixel_bary_coord_buffer,
	int* pixel_triangle_buffer, float* pixel_triangle_normals_buffer, int* indices_buffer, int* depth_locked, int width, int max_triangle_id)
{
	int triangle_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (triangle_index < max_triangle_id && triangles[triangle_index].should_draw) {
		TriangleToRasterize triangle = triangles[triangle_index];
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
							double green_ = alpha * static_cast<double>(triangle.v0.color.y) + beta * static_cast<double>(triangle.v1.color.y) + gamma * static_cast<double>(triangle.v2.color.y);
							double blue_ = alpha * static_cast<double>(triangle.v0.color.z) + beta * static_cast<double>(triangle.v1.color.z) + gamma * static_cast<double>(triangle.v2.color.z);

							// clamp bytes to 255
							const unsigned char red = static_cast<unsigned char>(255.0f * fminf(static_cast<float>(red_), 1.0f));
							const unsigned char green = static_cast<unsigned char>(255.0f * fminf(static_cast<float>(green_), 1.0f));
							const unsigned char blue = static_cast<unsigned char>(255.0f * fminf(static_cast<float>(blue_), 1.0f));

							// update buffers
							color_buffer[index * 3] = blue;
							color_buffer[index * 3 + 1] = green;
							color_buffer[index * 3 + 2] = red;

							pixel_bary_coord_buffer[index * 3] = alpha;
							pixel_bary_coord_buffer[index * 3 + 1] = beta;
							pixel_bary_coord_buffer[index * 3 + 2] = gamma;

							pixel_triangle_buffer[index] = triangle_index;

							int v_id_0 = indices_buffer[3 * triangle_index];
							int v_id_1 = indices_buffer[3 * triangle_index + 1];
							int v_id_2 = indices_buffer[3 * triangle_index + 2];

							for (int i = 0; i < 3; ++i) {
								pixel_triangle_normals_buffer[index * 9 + i] = vertex_normal_buffer[v_id_0 * 3 + i];
								pixel_triangle_normals_buffer[index * 9 + i + 3] = vertex_normal_buffer[v_id_1 * 3 + i];
								pixel_triangle_normals_buffer[index * 9 + i + 6] = vertex_normal_buffer[v_id_2 * 3 + i];
							}
						}

						if (isLocked) {
							depth_locked[index] = 0;
						}
					} while (!isLocked);
				}
			}
		}
	}
};

void Renderer::terminate_rendering_context() {
	// Free face parameter buffers
	cudaFreeAsync(device_vertices, streams[0]);
	cudaFreeAsync(device_vertex_normals, streams[1]);
	cudaFreeAsync(device_colors, streams[2]);
	cudaFreeAsync(device_triangles, streams[3]);

	// Free camera and illumination buffers
	cudaFreeAsync(device_mvp, streams[4]);
	cudaFreeAsync(device_mv, streams[5]);
	cudaFreeAsync(device_sh_coefficients, streams[6]);

	// Free rendered image and other information buffers
	cudaFreeAsync(device_rendered_color, streams[7]);
	cudaFreeAsync(device_pixel_bary_coord, streams[8]);
	cudaFreeAsync(device_pixel_triangle, streams[9]);
	cudaFreeAsync(device_depth, streams[10]);
	cudaFreeAsync(device_pixel_triangle_normals, streams[11]);

	// Free control buffer
	cudaFreeAsync(device_depth_locked, streams[12]);

	for (int i = 0; i < 13; ++i) {
		cudaStreamDestroy(streams[i]);
	}
}

void Renderer::clear_buffers() {
	cv::Mat depth = INT_MAX * cv::Mat::ones(viewport_height, viewport_width, CV_32SC1);

	cudaMemsetAsync(device_vertex_normals, 0, num_vertices * 3 * sizeof(float), streams[0]);

	cudaMemsetAsync(device_rendered_color, 0, viewport_height * viewport_width * 3 * sizeof(unsigned char), streams[1]);
	cudaMemsetAsync(device_pixel_bary_coord, 0, viewport_height * viewport_width * 3 * sizeof(double), streams[2]);
	cudaMemsetAsync(device_pixel_triangle, 0, viewport_height * viewport_width * sizeof(int), streams[3]);
	cudaMemcpyAsync(device_depth, depth.data, viewport_height * viewport_width * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemsetAsync(device_pixel_triangle_normals, 0, viewport_height * viewport_width * 9 * sizeof(float), streams[4]);
	cudaMemsetAsync(device_depth_locked, 0, viewport_height * viewport_width * sizeof(int), streams[5]);

	cudaDeviceSynchronize();
}

void Renderer::initialiaze_rendering_context(FaceModel& face_model, int height, int width) {
	viewport_height = height;
	viewport_width = width;

	// Initialize CPU buffers
	color_img = cv::Mat::zeros(viewport_height, viewport_width, CV_8UC3);
	depth_img = INT_MAX * cv::Mat::ones(viewport_height, viewport_width, CV_32SC1);
	pixel_bary_coord_buffer = cv::Mat::zeros(viewport_height, viewport_width, CV_64FC3);
	pixel_triangle_buffer = cv::Mat::zeros(viewport_height, viewport_width, CV_32S);
	pixel_triangle_normals_buffer = cv::Mat::zeros(viewport_height, viewport_width, CV_32FC(9));

	MatrixXi triangles = face_model.getTriangulation().transpose();

	num_vertices = face_model.getNumVertices();
	num_triangles = triangles.cols();
	re_rendered_vertex_color = VectorXf::Zero(num_vertices * 3);

	// Create CUDA strams
	for (int i = 0; i < 13; ++i) {
		cudaStreamCreate(&streams[i]);
	}

	// Allocate device memory for device buffers
	cudaMallocAsync((void**)&device_vertices, num_vertices * 3 * sizeof(float), streams[0]);
	cudaMallocAsync((void**)&device_vertex_normals, num_vertices * 3 * sizeof(float), streams[1]);
	cudaMallocAsync((void**)&device_colors, num_vertices * 3 * sizeof(float), streams[2]);
	cudaMallocAsync((void**)&device_triangles, num_triangles * 3 * sizeof(int), streams[3]);
	// Allocate device memory for camere and illumiation settings
	cudaMallocAsync((void**)&device_sh_coefficients, 27 * sizeof(float), streams[4]);
	cudaMallocAsync((void**)&device_mvp, 16 * sizeof(float), streams[5]);
	cudaMallocAsync((void**)&device_mv, 16 * sizeof(float), streams[6]);
	// Allocate device memory for control buffer
	cudaMallocAsync((void**)&device_depth_locked, viewport_height * viewport_width * sizeof(int), streams[7]);

	// Allocate device memory for rendered image and other necessary information
	cudaMallocAsync((void**)&device_rendered_color, viewport_height * viewport_width * 3 * sizeof(unsigned char), streams[8]);
	cudaMallocAsync((void**)&device_depth, viewport_height * viewport_width * sizeof(int), streams[9]);
	cudaMallocAsync((void**)&device_pixel_bary_coord, viewport_height * viewport_width * 3 * sizeof(double), streams[10]);
	cudaMallocAsync((void**)&device_pixel_triangle, viewport_height * viewport_width * sizeof(int), streams[11]);
	cudaMallocAsync((void**)&device_pixel_triangle_normals, viewport_height * viewport_width * 9 * sizeof(float), streams[12]);

	cudaDeviceSynchronize();

	// Initialiaze buffers
	cudaMemcpyAsync(device_triangles, triangles.data(), num_triangles * 3 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(device_depth, depth_img.data, viewport_height * viewport_width * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemsetAsync(device_rendered_color, 0, viewport_height * viewport_width * 3 * sizeof(unsigned char), streams[0]);
	cudaMemsetAsync(device_pixel_bary_coord, 0, viewport_height * viewport_width * 3 * sizeof(double), streams[1]);
	cudaMemsetAsync(device_pixel_triangle, 0, viewport_height * viewport_width * sizeof(int), streams[2]);
	cudaMemsetAsync(device_depth_locked, 0, viewport_height * viewport_width * sizeof(int), streams[3]);
	cudaMemsetAsync(device_pixel_triangle_normals, 0, viewport_height * viewport_width * 9 * sizeof(float), streams[4]);

	cudaDeviceSynchronize();
}

void Renderer::render(Matrix4f& mvp_matrix, Matrix4f& mv_matrix, VectorXf& vertices, VectorXf& colors, VectorXf& sh_red_coefficients,
	VectorXf& sh_green_coefficients, VectorXf& sh_blue_coefficients) {
	TriangleToRasterize* device_triangles_to_render;
	cudaMallocAsync((void**)&device_triangles_to_render, num_triangles * sizeof(TriangleToRasterize), streams[0]);
	cudaMemsetAsync(device_triangles_to_render, 0, num_triangles * sizeof(TriangleToRasterize), streams[0]);

	cudaMemcpyAsync(device_vertices, vertices.data(), num_vertices * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(device_colors, colors.data(), num_vertices * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(device_mvp, mvp_matrix.data(), 16 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(device_mv, mv_matrix.data(), 16 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(device_sh_coefficients, sh_red_coefficients.data(), 9 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(&device_sh_coefficients[9], sh_green_coefficients.data(), 9 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(&device_sh_coefficients[18], sh_blue_coefficients.data(), 9 * sizeof(float), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	int block_size = 256;
	int grid_size = int(num_triangles / block_size) + 1;

	// Compute vertex normals
	compute_vertex_normals <<<grid_size, block_size>>> (device_triangles, device_mv, device_vertices, device_vertex_normals, num_triangles);
	cudaDeviceSynchronize();

	// Compute vertex colors
	grid_size = int(num_vertices / block_size) + 1;
	compute_vertex_colors <<<grid_size, block_size>>> (device_colors, device_vertex_normals, device_sh_coefficients, num_vertices);
	cudaDeviceSynchronize();

	// Build triangles to rasterize
	grid_size = int(num_triangles / block_size) + 1;
	build_triangles <<<grid_size, block_size>>> (device_triangles, device_vertices, device_colors, device_mvp,
		device_triangles_to_render, num_triangles, viewport_width, viewport_height);
	cudaDeviceSynchronize();

	// Rasterize triangles
	raster_triangle <<<grid_size, block_size>>> (device_triangles_to_render, device_rendered_color, device_vertex_normals,
		device_depth, device_pixel_bary_coord, device_pixel_triangle, device_pixel_triangle_normals, device_triangles, device_depth_locked, 
		viewport_width, num_triangles);
	cudaDeviceSynchronize();

	cudaMemcpyAsync(color_img.data, device_rendered_color, viewport_height * viewport_width * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(depth_img.data, device_depth, viewport_height * viewport_width * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(pixel_bary_coord_buffer.data, device_pixel_bary_coord, viewport_height * viewport_width * 3 * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(pixel_triangle_buffer.data, device_pixel_triangle, viewport_height * viewport_width * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(pixel_triangle_normals_buffer.data, device_pixel_triangle_normals, viewport_height * viewport_width * 9 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(re_rendered_vertex_color.data(), device_colors, num_vertices * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
}