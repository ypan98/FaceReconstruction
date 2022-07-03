#pragma once
#include "Eigen.h"
#include "FaceModel.h"
#include "opencv2/core/core.hpp"
#include <chrono>

struct ModelVertex
{
	Vector4f position;
	Vector3f color;
};

struct TriangleToRasterize 
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

bool vertices_ccw_in_screen_space(const Vector4f& v0, const Vector4f& v1, const Vector4f& v2) {
	float dx01 = v1[0] - v0[0];
	float dy01 = v1[1] - v0[1];
	float dx02 = v2[0] - v0[0];
	float dy02 = v2[1] - v0[1];

	return (dx01 * dy02 - dy01 * dx02 < 0.0f);
}

Vector2f clip_to_screen_space(const Vector2f& clip_coords, int screen_width, int screen_height) {
	const float x_ss = (clip_coords[0] + 1.0f) * (screen_width / 2.0f);
	const float y_ss = screen_height - (clip_coords[1] + 1.0f) * (screen_height / 2.0f);
	return Vector2f(x_ss, y_ss);
}

bool are_vertices_ccw_in_screen_space(const Vector4f& v0, const Vector4f& v1, const Vector4f& v2)
{
	float dx01 = v1[0] - v0[0];
	float dy01 = v1[1] - v0[1];
	float dx02 = v2[0] - v0[0];
	float dy02 = v2[1] - v0[1];

	return (dx01 * dy02 - dy01 * dx02 < 0.0f);
};

cv::Rect calculate_clipped_bounding_box(Vector4f v0, Vector4f v1, Vector4f v2, int viewport_width, int viewport_height)
{
	using std::min;
	using std::max;
	using std::floor;
	using std::ceil;
	int minX = max(min(floor(v0[0]), min(floor(v1[0]), floor(v2[0]))), 0.0f);
	int maxX = min(max(ceil(v0[0]), max(ceil(v1[0]), ceil(v2[0]))), static_cast<float>(viewport_width - 1));
	int minY = max(min(floor(v0[1]), min(floor(v1[1]), floor(v2[1]))), 0.0f);
	int maxY = min(max(ceil(v0[1]), max(ceil(v1[1]), ceil(v2[1]))), static_cast<float>(viewport_height - 1));
	return cv::Rect(minX, minY, maxX - minX, maxY - minY);
};

std::pair<TriangleToRasterize, bool> process_prospective_tri(ModelVertex v0, ModelVertex v1, ModelVertex v2, int viewport_width, int viewport_height, bool enable_backface_culling)
{
	TriangleToRasterize t;

	t.v0 = v0;
	t.v1 = v1;
	t.v2 = v2;

	t.one_over_z0 = 1.0 / (double)t.v0.position[3];
	t.one_over_z1 = 1.0 / (double)t.v1.position[3];
	t.one_over_z2 = 1.0 / (double)t.v2.position[3];

	// divide by w
	// if ortho, we can do the divide as well, it will just be a / 1.0f.
	t.v0.position = t.v0.position / t.v0.position[3];
	t.v1.position = t.v1.position / t.v1.position[3];
	t.v2.position = t.v2.position / t.v2.position[3];

	Vector2f v0_screen = clip_to_screen_space(Vector2f(t.v0.position[0], t.v0.position[1]), viewport_width, viewport_height);
	t.v0.position[0] = v0_screen[0];
	t.v0.position[1] = v0_screen[1];
	Vector2f v1_screen = clip_to_screen_space(Vector2f(t.v1.position[0], t.v1.position[1]), viewport_width, viewport_height);
	t.v1.position[0] = v1_screen[0];
	t.v1.position[1] = v1_screen[1];
	Vector2f v2_screen = clip_to_screen_space(Vector2f(t.v2.position[0], t.v2.position[1]), viewport_width, viewport_height);
	t.v2.position[0] = v2_screen[0];
	t.v2.position[1] = v2_screen[1];

	if (enable_backface_culling) {
		if (!are_vertices_ccw_in_screen_space(t.v0.position, t.v1.position, t.v2.position))
			return std::make_pair(t, false);
	}

	// Get the bounding box of the triangle:
	cv::Rect boundingBox = calculate_clipped_bounding_box(t.v0.position, t.v1.position, t.v2.position, viewport_width, viewport_height);
	t.min_x = boundingBox.x;
	t.max_x = boundingBox.x + boundingBox.width;
	t.min_y = boundingBox.y;
	t.max_y = boundingBox.y + boundingBox.height;

	if (t.max_x <= t.min_x || t.max_y <= t.min_y)
		return std::make_pair(t, false);

	return std::make_pair(t, true);
};

std::vector<ModelVertex> clip_polygon_to_plane_in_4d(const std::vector<ModelVertex>& vertices, const Vector4f& plane_normal)
{
	std::vector<ModelVertex> clippedVertices;

	// We can have 2 cases:
	//	* 1 ModelVertex visible: we make 1 new triangle out of the visible ModelVertex plus the 2 intersection points with the near-plane
	//  * 2 vertices visible: we have a quad, so we have to make 2 new triangles out of it.

	for (unsigned int i = 0; i < vertices.size(); i++)
	{
		int a = i; // the current ModelVertex
		int b = (i + 1) % vertices.size(); // the following ModelVertex (wraps around 0)

		float fa = vertices[a].position.dot(plane_normal); // Note: Shouldn't they be unit length?
		float fb = vertices[b].position.dot(plane_normal); // < 0 means on visible side, > 0 means on invisible side?

		if ((fa < 0 && fb > 0) || (fa > 0 && fb < 0)) // one ModelVertex is on the visible side of the plane, one on the invisible? so we need to split?
		{
			Vector4f direction = vertices[b].position - vertices[a].position;
			float t = -(plane_normal.dot(vertices[a].position)) / (plane_normal.dot(direction)); // the parametric value on the line, where the line to draw intersects the plane?

			// generate a new ModelVertex at the line-plane intersection point
			Vector4f position = vertices[a].position + t * direction;
			Vector3f color = vertices[a].color + t * (vertices[b].color - vertices[a].color);

			ModelVertex vertex;
			vertex.position = position;
			vertex.color = color;

			if (fa < 0) // we keep the original vertex plus the new one
			{
				clippedVertices.push_back(vertices[a]);
				clippedVertices.push_back(vertex);
			}
			else if (fb < 0) // we use only the new vertex
			{
				clippedVertices.push_back(vertex);
			}
		}
		else if (fa < 0 && fb < 0) // both are visible (on the "good" side of the plane), no splitting required, use the current vertex
		{
			clippedVertices.push_back(vertices[a]);
		}
		// else, both vertices are not visible, nothing to add and draw
	}

	return clippedVertices;
};

double implicit_line(float x, float y, const Vector4f& v1, const Vector4f& v2)
{
	return ((double)v1[1] - (double)v2[1]) * (double)x + ((double)v2[0] - (double)v1[0]) * (double)y + (double)v1[0] * (double)v2[1] - (double)v2[0] * (double)v1[1];
};

void raster_triangle(TriangleToRasterize triangle, cv::Mat img, cv::Mat depthbuffer, bool enable_far_clipping)
{
	for (int yi = triangle.min_y; yi <= triangle.max_y; ++yi)
	{
		for (int xi = triangle.min_x; xi <= triangle.max_x; ++xi)
		{
			const float x = static_cast<float>(xi) + 0.5f;
			const float y = static_cast<float>(yi) + 0.5f;

			// these will be used for barycentric weights computation
			const double one_over_v0ToLine12 = 1.0 / implicit_line(triangle.v0.position[0], triangle.v0.position[1], triangle.v1.position, triangle.v2.position);
			const double one_over_v1ToLine20 = 1.0 / implicit_line(triangle.v1.position[0], triangle.v1.position[1], triangle.v2.position, triangle.v0.position);
			const double one_over_v2ToLine01 = 1.0 / implicit_line(triangle.v2.position[0], triangle.v2.position[1], triangle.v0.position, triangle.v1.position);
			// affine barycentric weights
			double alpha = implicit_line(x, y, triangle.v1.position, triangle.v2.position) * one_over_v0ToLine12;
			double beta = implicit_line(x, y, triangle.v2.position, triangle.v0.position) * one_over_v1ToLine20;
			double gamma = implicit_line(x, y, triangle.v0.position, triangle.v1.position) * one_over_v2ToLine01;

			// if pixel (x, y) is inside the triangle or on one of its edges
			if (alpha >= 0 && beta >= 0 && gamma >= 0)
			{
				const int pixel_index_row = yi;
				const int pixel_index_col = xi;

				const double z_affine = alpha * static_cast<double>(triangle.v0.position[2]) + beta * static_cast<double>(triangle.v1.position[2]) + gamma * static_cast<double>(triangle.v2.position[2]);
				// The '<= 1.0' clips against the far-plane in NDC. We clip against the near-plane earlier.
				bool draw = true;
				if (enable_far_clipping)
				{
					if (z_affine > 1.0)
					{
						draw = false;
					}
				}

				if (z_affine < depthbuffer.at<double>(pixel_index_row, pixel_index_col) && draw)
				{
					// perspective-correct barycentric weights
					double d = alpha * triangle.one_over_z0 + beta * triangle.one_over_z1 + gamma * triangle.one_over_z2;
					d = 1.0 / d;
					alpha *= d * triangle.one_over_z0; // In case of affine cam matrix, everything is 1 and a/b/g don't get changed.
					beta *= d * triangle.one_over_z1;
					gamma *= d * triangle.one_over_z2;

					// attributes interpolation
					Vector3f pixel_color = alpha * triangle.v0.color + beta * triangle.v1.color + gamma * triangle.v2.color;

					// clamp bytes to 255
					const unsigned char red = static_cast<unsigned char>(255.0f * std::min(pixel_color[0], 1.0f));
					const unsigned char green = static_cast<unsigned char>(255.0f * std::min(pixel_color[1], 1.0f));
					const unsigned char blue = static_cast<unsigned char>(255.0f * std::min(pixel_color[2], 1.0f));

					// update buffers
					img.at<cv::Vec4b>(pixel_index_row, pixel_index_col)[0] = blue;
					img.at<cv::Vec4b>(pixel_index_row, pixel_index_col)[1] = green;
					img.at<cv::Vec4b>(pixel_index_row, pixel_index_col)[2] = red;
					img.at<cv::Vec4b>(pixel_index_row, pixel_index_col)[3] = 255; // alpha channel
					depthbuffer.at<double>(pixel_index_row, pixel_index_col) = z_affine;
				}
			}
		}
	}
};

cv::Mat render(FaceModel face_model, Matrix4f projection_matrix, int viewport_width, int viewport_height, bool enable_backface_culling = true, bool enable_near_clipping = true, bool enable_far_clipping = true)
{
	auto start = std::chrono::steady_clock::now();
	cv::Mat img = cv::Mat::zeros(viewport_height, viewport_width, CV_8UC4); // make sure it's CV_8UC4?
	cv::Mat depthbuffer = std::numeric_limits<float>::max() * cv::Mat::ones(viewport_height, viewport_width, CV_64FC1);

	// Get the coordinate of the face model's vertices in clipspace
	MatrixXf model_vertices(4, face_model.getNumVertices());
	model_vertices.block(0, 0, 3, face_model.getNumVertices()) = face_model.getShapeMean().reshaped(3, face_model.getNumVertices());
	model_vertices.row(3) = VectorXf::Ones(face_model.getNumVertices());
	MatrixXf clipspace_vertices_coord = (projection_matrix * model_vertices).transpose();
	auto end = std::chrono::steady_clock::now();
	std::cout << "time used for calculating clipspace coords: " << std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) << "ms\n";

	start = std::chrono::steady_clock::now();
	std::vector<TriangleToRasterize> triangles_to_raster;


	for (int i = 0; i < face_model.getTriangulation().rows(); ++i) {
		Vector3i tri_indices = face_model.getTriangulation().row(i);

		// Get the vertices of this triangle
		ModelVertex v0, v1, v2;
		v0.position = clipspace_vertices_coord.row(tri_indices[0]);
		v0.color = Vector3f(face_model.getColorMean()[tri_indices[0] * 3], face_model.getColorMean()[tri_indices[0] * 3 + 1], 
			face_model.getColorMean()[tri_indices[0] * 3 + 2]);

		v1.position = clipspace_vertices_coord.row(tri_indices[1]);
		v1.color = Vector3f(face_model.getColorMean()[tri_indices[1] * 3], face_model.getColorMean()[tri_indices[1] * 3 + 1],
			face_model.getColorMean()[tri_indices[1] * 3 + 2]);

		v2.position = clipspace_vertices_coord.row(tri_indices[2]);
		v2.color = Vector3f(face_model.getColorMean()[tri_indices[2] * 3], face_model.getColorMean()[tri_indices[2] * 3 + 1],
			face_model.getColorMean()[tri_indices[2] * 3 + 2]);

		std::vector<ModelVertex> vertices{ v0, v1, v2 };

		// Get the visibility of each vertex
		unsigned char visibility_bits[3];
		for (unsigned char k = 0; k < 3; k++)
		{
			visibility_bits[k] = 0;

			float x_cc = clipspace_vertices_coord.row(tri_indices[k])[0];
			float y_cc = clipspace_vertices_coord.row(tri_indices[k])[1];
			float z_cc = clipspace_vertices_coord.row(tri_indices[k])[2];
			float w_cc = clipspace_vertices_coord.row(tri_indices[k])[3];

			if (x_cc < -w_cc)			// true if outside of view frustum. False if on or inside the plane.
				visibility_bits[k] |= 1;	// set bit if outside of frustum
			if (x_cc > w_cc)
				visibility_bits[k] |= 2;
			if (y_cc < -w_cc)
				visibility_bits[k] |= 4;
			if (y_cc > w_cc)
				visibility_bits[k] |= 8;
			if (enable_near_clipping && z_cc < -w_cc) // near plane frustum clipping
				visibility_bits[k] |= 16;
			if (enable_far_clipping && z_cc > w_cc) // far plane frustum clipping
				visibility_bits[k] |= 32;
		} // if all bits are 0, then it's inside the frustum
		// all vertices are not visible - reject the triangle.
		if ((visibility_bits[0] & visibility_bits[1] & visibility_bits[2]) > 0)
		{
			continue;
		}
		// all vertices are visible - pass the whole triangle to the rasterizer. = All bits of all 3 triangles are 0.
		if ((visibility_bits[0] | visibility_bits[1] | visibility_bits[2]) == 0)
		{

			std::pair t = process_prospective_tri(vertices[0], vertices[1], vertices[2], viewport_width, viewport_height, enable_backface_culling);
			if (t.second) {
				raster_triangle(t.first, img, depthbuffer, enable_far_clipping);
			}
			continue;
		}

		// split the triangle if it intersects the near plane:
		if (enable_near_clipping)
		{
			vertices = clip_polygon_to_plane_in_4d(vertices, Vector4f(0.0f, 0.0f, -1.0f, -1.0f));
		}

		// triangulation of the polygon formed of vertices array
		if (vertices.size() >= 3)
		{
			for (unsigned char k = 0; k < vertices.size() - 2; k++)
			{
				std::pair t = process_prospective_tri(vertices[0], vertices[1 + k], vertices[2 + k], viewport_width, viewport_height, enable_backface_culling);
				if (t.second) {
					raster_triangle(t.first, img, depthbuffer, enable_far_clipping);
				}
			}
		}
	}
	end = std::chrono::steady_clock::now();
	std::cout << "time used for calculating rest_triangles: " << std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) << "ms\n";

	start = std::chrono::steady_clock::now();
	// Fragment/pixel shader: Colour the pixel values
	// for every tri:
	for (const auto& tri : triangles_to_raster) {
		raster_triangle(tri, img, depthbuffer, enable_far_clipping);
	}
	end = std::chrono::steady_clock::now();
	std::cout << "time used for vertex/fragment shader: " << std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) << "ms\n";

	return img;
};