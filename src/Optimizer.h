#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "Face.h"
#include "PoseIncrement.h"
#include "Projection.h"
#include <stdio.h>
#include <omp.h>
#include "Renderer.h"
#include <opencv2/core/core.hpp>

using namespace std;

template <typename T>
T implicit_line(T x, T y, const Matrix<T, 4, 1>& v1, const Matrix<T, 4, 1>& v2)
{
    return (v1(1, 0) - v2(1, 0)) * x + (v2(0, 0) - v1(0, 0)) * y + v1(0, 0) * v2(1, 0) - v2(0, 0) * v1(1, 0);
};

double SH_basis_function(Vector3d& normal, int basis_index) {
    switch (basis_index)
    {
    case 0:
        return 0.282095 * 3.1415926;
    case 1:
        return -0.488603 * normal(1) * 2.094395;
    case 2:
        return 0.488603 * normal(2) * 2.094395;
    case 3:
        return -0.488603 * normal(0) * 2.094395;
    case 4:
        return 1.092548 * normal(0) * normal(1) * 0.785398;
    case 5:
        return -1.092548 * normal(1) * normal(2) * 0.785398;
    case 6:
        return 0.315392 * (3. * normal(2) * normal(2) - 1.) * 0.785398;
    case 7:
        return -1.092548 * normal(0) * normal(2) * 0.785398;
    case 8:
        return 0.546274 * (normal(0) * normal(0) - normal(1) * normal(1)) * 0.785398;
    default:
        return 0.;
    }
};

//--------------------------------------------------Energy Terms Used for Neutral Face Reconstruction From Single Image--------------------------------------------------//
/*
* Feature Similarity (Landmarks) energy optimization.
* Truly denser problem: every residual depends on every parameter alpha, beta, extrinsic and intrinsic. => No big difference defining a residual block per point or
* the whole as a big block. By means of code consistency we still do it by point.
*/
class FeatureSimilarityEnergy {
public:
    FeatureSimilarityEnergy(const double& _landmarkWeight, const Vector2d& _landmark, const FaceModel* _faceModel, const unsigned& _vertexIdx, Matrix4d _perspective_matrix,
        const unsigned& _viewport_width, const unsigned& _viewport_height) :
        landmarkWeight(_landmarkWeight),
        landmark(_landmark),
        faceModel(_faceModel),
        vertexIdx(_vertexIdx),
        perspective_matrix(_perspective_matrix),
        viewport_width(_viewport_width),
        viewport_height(_viewport_height)
    { }

    template <typename T>
    bool operator()(const T const* alpha, const T const* gamma, const T* const extrinsicsArr, T* residuals) const {
        Map<const Matrix<T, -1, 1> > alphaMap(alpha, BFM_ALPHA_SIZE);
        Map<const Matrix<T, -1, 1> > gammaMap(gamma, BFM_GAMMA_SIZE);
        // calculate vertex
        Matrix<T, -1, -1> vertex = (*faceModel).getShapeMeanBlock(3 * vertexIdx, 3).cast<T>()
            + (*faceModel).getExpMeanBlock(3 * vertexIdx, 3).cast<T>()
            + (*faceModel).getShapeBasisRowBlock(3 * vertexIdx, 3) * alphaMap
            + (*faceModel).getExpBasisRowBlock(3 * vertexIdx, 3) * gammaMap;

        // apply pose (extrinsics)
        T vertex_transformed[3];
        PoseIncrement<T> pose_inc(const_cast<T* const>(extrinsicsArr));
        pose_inc.apply(vertex.data(), vertex_transformed);
        // apply perspective projection (instrinsics)
        Matrix<T, 4, 1> vertex_homogeneous;
        vertex_homogeneous << vertex_transformed[0], vertex_transformed[1], vertex_transformed[2], T(1);
        Matrix<T, -1, -1> vertex_clip_space = perspective_matrix * vertex_homogeneous;
        T x_pixel_space = (vertex_clip_space(0, 0) / vertex_clip_space(3, 0) + T(1)) * (T(viewport_width) / T(2));
        T y_pixel_space = T(viewport_height) - (vertex_clip_space(1, 0) / vertex_clip_space(3, 0) + T(1)) * (T(viewport_height) / T(2));
        residuals[0] = (T(landmark(0)) - x_pixel_space)*T(landmarkWeight);
        residuals[1] = (T(landmark(1)) - y_pixel_space)*T(landmarkWeight);
        return true;
    }

private:
    const double landmarkWeight;
    const FaceModel* faceModel;
    const Vector2d landmark;
    const unsigned vertexIdx, viewport_width, viewport_height;
    Matrix4d perspective_matrix;
};

class GeometryConsistencyEnergy {
public:
    GeometryConsistencyEnergy(const double& _pointWeight, const double& _planeWeight, const Vector3i& _vertex_indices_current_point, const Vector3i& _vertex_indices_top, const Vector3i& _vertex_indices_down,
        const Vector3i& _vertex_indices_left, const Vector3i& _vertex_indices_right, const double& _depth_source, const Vector3d& _point_normal_source, const FaceModel* _faceModel,
        const Matrix4d& _perspective_matrix, const double& _pixel_row, const double& _pixel_col, const double& _width, const double& _height, const double& _z_near, const double& _z_far) :
        pointWeight(_pointWeight),
        planeWeight(_planeWeight),
        vertex_indices_current_point(_vertex_indices_current_point),
        vertex_indices_top(_vertex_indices_top),
        vertex_indices_down(_vertex_indices_down),
        vertex_indices_left(_vertex_indices_left),
        vertex_indices_right(_vertex_indices_right),
        point_normal_source(_point_normal_source),
        depth_source(_depth_source),
        faceModel(_faceModel),
        perspective_matrix(_perspective_matrix),
        pixel_row(_pixel_row),
        pixel_col(_pixel_col),
        width(_width),
        height(_height),
        z_near(_z_near),
        z_far(_z_far)
    { }

    template <typename T>
    bool operator() (const T const* alpha, const T const* gamma, const T* const extrinsicsArr, T* residuals) const {
        Map<const Matrix<T, -1, 1> > alphaMap(alpha, BFM_ALPHA_SIZE);
        Map<const Matrix<T, -1, 1> > gammaMap(gamma, BFM_GAMMA_SIZE);

        //     Calculate vertex
        Matrix<T, -1, -1> vertex_0 = (*faceModel).getShapeMeanBlock(3 * vertex_indices_current_point(0), 3).cast<T>()
            + (*faceModel).getExpMeanBlock(3 * vertex_indices_current_point(0), 3).cast<T>()
            + (*faceModel).getShapeBasisRowBlock(3 * vertex_indices_current_point(0), 3) * alphaMap
            + (*faceModel).getExpBasisRowBlock(3 * vertex_indices_current_point(0), 3) * gammaMap;

        Matrix<T, -1, -1> vertex_1 = (*faceModel).getShapeMeanBlock(3 * vertex_indices_current_point(1), 3).cast<T>()
            + (*faceModel).getExpMeanBlock(3 * vertex_indices_current_point(1), 3).cast<T>()
            + (*faceModel).getShapeBasisRowBlock(3 * vertex_indices_current_point(1), 3) * alphaMap
            + (*faceModel).getExpBasisRowBlock(3 * vertex_indices_current_point(1), 3) * gammaMap;

        Matrix<T, -1, -1> vertex_2 = (*faceModel).getShapeMeanBlock(3 * vertex_indices_current_point(2), 3).cast<T>()
            + (*faceModel).getExpMeanBlock(3 * vertex_indices_current_point(2), 3).cast<T>()
            + (*faceModel).getShapeBasisRowBlock(3 * vertex_indices_current_point(2), 3) * alphaMap
            + (*faceModel).getExpBasisRowBlock(3 * vertex_indices_current_point(2), 3) * gammaMap;

        //     apply pose (extrinsics)
        PoseIncrement<T> pose_inc(const_cast<T* const>(extrinsicsArr));
        pose_inc.apply(vertex_0.data(), vertex_0.data());
        pose_inc.apply(vertex_1.data(), vertex_1.data());
        pose_inc.apply(vertex_2.data(), vertex_2.data());

        Matrix<T, 4, 1> vertex_0_screen;
        vertex_0_screen << vertex_0(0), vertex_0(1), vertex_0(2), T(1);
        Matrix<T, 4, 1> vertex_1_screen;
        vertex_1_screen << vertex_1(0), vertex_1(1), vertex_1(2), T(1);
        Matrix<T, 4, 1> vertex_2_screen;
        vertex_2_screen << vertex_2(0), vertex_2(1), vertex_2(2), T(1);

        //      apply perspective projection (instrinsics)
        vertex_0_screen = perspective_matrix * vertex_0_screen;
        vertex_1_screen = perspective_matrix * vertex_1_screen;
        vertex_2_screen = perspective_matrix * vertex_2_screen;

        vertex_0_screen = vertex_0_screen / vertex_0_screen(3, 0);
        vertex_1_screen = vertex_1_screen / vertex_1_screen(3, 0);
        vertex_2_screen = vertex_2_screen / vertex_2_screen(3, 0);

        //      To pixel space
        vertex_0_screen(0, 0) = (vertex_0_screen(0, 0) + T(1)) * (T(width) / T(2));
        vertex_0_screen(1, 0) = T(height) - (vertex_0_screen(1, 0) + T(1)) * (T(height) / T(2));
        vertex_1_screen(0, 0) = (vertex_1_screen(0, 0) + T(1)) * (T(width) / T(2));
        vertex_1_screen(1, 0) = T(height) - (vertex_1_screen(1, 0) + T(1)) * (T(height) / T(2));
        vertex_2_screen(0, 0) = (vertex_2_screen(0, 0) + T(1)) * (T(width) / T(2));
        vertex_2_screen(1, 0) = T(height) - (vertex_2_screen(1, 0) + T(1)) * (T(height) / T(2));

        T one_over_v0ToLine12 = T(1) / implicit_line(vertex_0_screen(0, 0), vertex_0_screen(1, 0), vertex_1_screen, vertex_2_screen);
        T one_over_v1ToLine20 = T(1) / implicit_line(vertex_1_screen(0, 0), vertex_1_screen(1, 0), vertex_2_screen, vertex_0_screen);
        T one_over_v2ToLine01 = T(1) / implicit_line(vertex_2_screen(0, 0), vertex_2_screen(1, 0), vertex_0_screen, vertex_1_screen);

        T alpha_bary = implicit_line(T(pixel_col), T(pixel_row), vertex_1_screen, vertex_2_screen) * one_over_v0ToLine12;
        T beta_bary = implicit_line(T(pixel_col), T(pixel_row), vertex_2_screen, vertex_0_screen) * one_over_v1ToLine20;
        T gamma_bary = implicit_line(T(pixel_col), T(pixel_row), vertex_0_screen, vertex_1_screen) * one_over_v2ToLine01;

        T vertex0_z = alpha_bary * vertex_0_screen(2, 0) + beta_bary * vertex_1_screen(2, 0) + gamma_bary * vertex_2_screen(2, 0);

        vertex0_z = T(1) - ((T(2) * T(z_near) * T(z_far) / (T(z_far) + T(z_near) - vertex0_z * (T(z_far) - T(z_near)))) - T(z_near)) / (T(z_far) - T(z_near));

        // Point to point constraint
        residuals[0] = (vertex0_z - depth_source) * T(pointWeight);

        Matrix<T, -1, -1> vertex_top_0 = (*faceModel).getShapeMeanBlock(3 * vertex_indices_top(0), 3).cast<T>()
            + (*faceModel).getExpMeanBlock(3 * vertex_indices_top(0), 3).cast<T>()
            + (*faceModel).getShapeBasisRowBlock(3 * vertex_indices_top(0), 3) * alphaMap
            + (*faceModel).getExpBasisRowBlock(3 * vertex_indices_top(0), 3) * gammaMap;

        Matrix<T, -1, -1> vertex_top_1 = (*faceModel).getShapeMeanBlock(3 * vertex_indices_top(1), 3).cast<T>()
            + (*faceModel).getExpMeanBlock(3 * vertex_indices_top(1), 3).cast<T>()
            + (*faceModel).getShapeBasisRowBlock(3 * vertex_indices_top(1), 3) * alphaMap
            + (*faceModel).getExpBasisRowBlock(3 * vertex_indices_top(1), 3) * gammaMap;

        Matrix<T, -1, -1> vertex_top_2 = (*faceModel).getShapeMeanBlock(3 * vertex_indices_top(2), 3).cast<T>()
            + (*faceModel).getExpMeanBlock(3 * vertex_indices_top(2), 3).cast<T>()
            + (*faceModel).getShapeBasisRowBlock(3 * vertex_indices_top(2), 3) * alphaMap
            + (*faceModel).getExpBasisRowBlock(3 * vertex_indices_top(2), 3) * gammaMap;

        Matrix<T, -1, -1> vertex_down_0 = (*faceModel).getShapeMeanBlock(3 * vertex_indices_down(0), 3).cast<T>()
            + (*faceModel).getExpMeanBlock(3 * vertex_indices_down(0), 3).cast<T>()
            + (*faceModel).getShapeBasisRowBlock(3 * vertex_indices_down(0), 3) * alphaMap
            + (*faceModel).getExpBasisRowBlock(3 * vertex_indices_down(0), 3) * gammaMap;

        Matrix<T, -1, -1> vertex_down_1 = (*faceModel).getShapeMeanBlock(3 * vertex_indices_down(1), 3).cast<T>()
            + (*faceModel).getExpMeanBlock(3 * vertex_indices_down(1), 3).cast<T>()
            + (*faceModel).getShapeBasisRowBlock(3 * vertex_indices_down(1), 3) * alphaMap
            + (*faceModel).getExpBasisRowBlock(3 * vertex_indices_down(1), 3) * gammaMap;

        Matrix<T, -1, -1> vertex_down_2 = (*faceModel).getShapeMeanBlock(3 * vertex_indices_down(2), 3).cast<T>()
            + (*faceModel).getExpMeanBlock(3 * vertex_indices_down(2), 3).cast<T>()
            + (*faceModel).getShapeBasisRowBlock(3 * vertex_indices_down(2), 3) * alphaMap
            + (*faceModel).getExpBasisRowBlock(3 * vertex_indices_down(2), 3) * gammaMap;

        Matrix<T, -1, -1> vertex_left_0 = (*faceModel).getShapeMeanBlock(3 * vertex_indices_left(0), 3).cast<T>()
            + (*faceModel).getExpMeanBlock(3 * vertex_indices_left(0), 3).cast<T>()
            + (*faceModel).getShapeBasisRowBlock(3 * vertex_indices_left(0), 3) * alphaMap
            + (*faceModel).getExpBasisRowBlock(3 * vertex_indices_left(0), 3) * gammaMap;

        Matrix<T, -1, -1> vertex_left_1 = (*faceModel).getShapeMeanBlock(3 * vertex_indices_left(1), 3).cast<T>()
            + (*faceModel).getExpMeanBlock(3 * vertex_indices_left(1), 3).cast<T>()
            + (*faceModel).getShapeBasisRowBlock(3 * vertex_indices_left(1), 3) * alphaMap
            + (*faceModel).getExpBasisRowBlock(3 * vertex_indices_left(1), 3) * gammaMap;

        Matrix<T, -1, -1> vertex_left_2 = (*faceModel).getShapeMeanBlock(3 * vertex_indices_left(2), 3).cast<T>()
            + (*faceModel).getExpMeanBlock(3 * vertex_indices_left(2), 3).cast<T>()
            + (*faceModel).getShapeBasisRowBlock(3 * vertex_indices_left(2), 3) * alphaMap
            + (*faceModel).getExpBasisRowBlock(3 * vertex_indices_left(2), 3) * gammaMap;

        Matrix<T, -1, -1> vertex_right_0 = (*faceModel).getShapeMeanBlock(3 * vertex_indices_right(0), 3).cast<T>()
            + (*faceModel).getExpMeanBlock(3 * vertex_indices_right(0), 3).cast<T>()
            + (*faceModel).getShapeBasisRowBlock(3 * vertex_indices_right(0), 3) * alphaMap
            + (*faceModel).getExpBasisRowBlock(3 * vertex_indices_right(0), 3) * gammaMap;

        Matrix<T, -1, -1> vertex_right_1 = (*faceModel).getShapeMeanBlock(3 * vertex_indices_right(1), 3).cast<T>()
            + (*faceModel).getExpMeanBlock(3 * vertex_indices_right(1), 3).cast<T>()
            + (*faceModel).getShapeBasisRowBlock(3 * vertex_indices_right(1), 3) * alphaMap
            + (*faceModel).getExpBasisRowBlock(3 * vertex_indices_right(1), 3) * gammaMap;

        Matrix<T, -1, -1> vertex_right_2 = (*faceModel).getShapeMeanBlock(3 * vertex_indices_right(2), 3).cast<T>()
            + (*faceModel).getExpMeanBlock(3 * vertex_indices_right(2), 3).cast<T>()
            + (*faceModel).getShapeBasisRowBlock(3 * vertex_indices_right(2), 3) * alphaMap
            + (*faceModel).getExpBasisRowBlock(3 * vertex_indices_right(2), 3) * gammaMap;

        pose_inc.apply(vertex_top_0.data(), vertex_top_0.data());
        pose_inc.apply(vertex_top_1.data(), vertex_top_1.data());
        pose_inc.apply(vertex_top_2.data(), vertex_top_2.data());

        pose_inc.apply(vertex_down_0.data(), vertex_down_0.data());
        pose_inc.apply(vertex_down_1.data(), vertex_down_1.data());
        pose_inc.apply(vertex_down_2.data(), vertex_down_2.data());

        pose_inc.apply(vertex_left_0.data(), vertex_left_0.data());
        pose_inc.apply(vertex_left_1.data(), vertex_left_1.data());
        pose_inc.apply(vertex_left_2.data(), vertex_left_2.data());

        pose_inc.apply(vertex_right_0.data(), vertex_right_0.data());
        pose_inc.apply(vertex_right_1.data(), vertex_right_1.data());
        pose_inc.apply(vertex_right_2.data(), vertex_right_2.data());

        Matrix<T, 4, 1> vertex_top_0_screen;
        vertex_top_0_screen << vertex_top_0(0), vertex_top_0(1), vertex_top_0(2), T(1);
        Matrix<T, 4, 1> vertex_top_1_screen;
        vertex_top_1_screen << vertex_top_1(0), vertex_top_1(1), vertex_top_1(2), T(1);
        Matrix<T, 4, 1> vertex_top_2_screen;
        vertex_top_2_screen << vertex_top_2(0), vertex_top_2(1), vertex_top_2(2), T(1);

        Matrix<T, 4, 1> vertex_down_0_screen;
        vertex_down_0_screen << vertex_down_0(0), vertex_down_0(1), vertex_down_0(2), T(1);
        Matrix<T, 4, 1> vertex_down_1_screen;
        vertex_down_1_screen << vertex_down_1(0), vertex_down_1(1), vertex_down_1(2), T(1);
        Matrix<T, 4, 1> vertex_down_2_screen;
        vertex_down_2_screen << vertex_down_2(0), vertex_down_2(1), vertex_down_2(2), T(1);

        Matrix<T, 4, 1> vertex_left_0_screen;
        vertex_left_0_screen << vertex_left_0(0), vertex_left_0(1), vertex_left_0(2), T(1);
        Matrix<T, 4, 1> vertex_left_1_screen;
        vertex_left_1_screen << vertex_left_1(0), vertex_left_1(1), vertex_left_1(2), T(1);
        Matrix<T, 4, 1> vertex_left_2_screen;
        vertex_left_2_screen << vertex_left_2(0), vertex_left_2(1), vertex_left_2(2), T(1);

        Matrix<T, 4, 1> vertex_right_0_screen;
        vertex_right_0_screen << vertex_right_0(0), vertex_right_0(1), vertex_right_0(2), T(1);
        Matrix<T, 4, 1> vertex_right_1_screen;
        vertex_right_1_screen << vertex_right_1(0), vertex_right_1(1), vertex_right_1(2), T(1);
        Matrix<T, 4, 1> vertex_right_2_screen;
        vertex_right_2_screen << vertex_right_2(0), vertex_right_2(1), vertex_right_2(2), T(1);
        
        //      apply perspective projection (instrinsics)
        vertex_top_0_screen = perspective_matrix * vertex_top_0_screen;
        vertex_top_1_screen = perspective_matrix * vertex_top_1_screen;
        vertex_top_2_screen = perspective_matrix * vertex_top_2_screen;

        vertex_down_0_screen = perspective_matrix * vertex_down_0_screen;
        vertex_down_1_screen = perspective_matrix * vertex_down_1_screen;
        vertex_down_2_screen = perspective_matrix * vertex_down_2_screen;

        vertex_left_0_screen = perspective_matrix * vertex_left_0_screen;
        vertex_left_1_screen = perspective_matrix * vertex_left_1_screen;
        vertex_left_2_screen = perspective_matrix * vertex_left_2_screen;

        vertex_right_0_screen = perspective_matrix * vertex_right_0_screen;
        vertex_right_1_screen = perspective_matrix * vertex_right_1_screen;
        vertex_right_2_screen = perspective_matrix * vertex_right_2_screen;

        vertex_top_0_screen = vertex_top_0_screen / vertex_top_0_screen(3, 0);
        vertex_top_1_screen = vertex_top_1_screen / vertex_top_1_screen(3, 0);
        vertex_top_2_screen = vertex_top_2_screen / vertex_top_2_screen(3, 0);

        vertex_down_0_screen = vertex_down_0_screen / vertex_down_0_screen(3, 0);
        vertex_down_1_screen = vertex_down_1_screen / vertex_down_1_screen(3, 0);
        vertex_down_2_screen = vertex_down_2_screen / vertex_down_2_screen(3, 0);

        vertex_left_0_screen = vertex_left_0_screen / vertex_left_0_screen(3, 0);
        vertex_left_1_screen = vertex_left_1_screen / vertex_left_1_screen(3, 0);
        vertex_left_2_screen = vertex_left_2_screen / vertex_left_2_screen(3, 0);

        vertex_right_0_screen = vertex_right_0_screen / vertex_right_0_screen(3, 0);
        vertex_right_1_screen = vertex_right_1_screen / vertex_right_1_screen(3, 0);
        vertex_right_2_screen = vertex_right_2_screen / vertex_right_2_screen(3, 0);

        //      To pixel space
        vertex_top_0_screen(0, 0) = (vertex_top_0_screen(0, 0) + T(1)) * (T(width) / T(2));
        vertex_top_0_screen(1, 0) = T(height) - (vertex_top_0_screen(1, 0) + T(1)) * (T(height) / T(2));
        vertex_top_1_screen(0, 0) = (vertex_top_1_screen(0, 0) + T(1)) * (T(width) / T(2));
        vertex_top_1_screen(1, 0) = T(height) - (vertex_top_1_screen(1, 0) + T(1)) * (T(height) / T(2));
        vertex_top_2_screen(0, 0) = (vertex_top_2_screen(0, 0) + T(1)) * (T(width) / T(2));
        vertex_top_2_screen(1, 0) = T(height) - (vertex_top_2_screen(1, 0) + T(1)) * (T(height) / T(2));

        vertex_down_0_screen(0, 0) = (vertex_down_0_screen(0, 0) + T(1)) * (T(width) / T(2));
        vertex_down_0_screen(1, 0) = T(height) - (vertex_down_0_screen(1, 0) + T(1)) * (T(height) / T(2));
        vertex_down_1_screen(0, 0) = (vertex_down_1_screen(0, 0) + T(1)) * (T(width) / T(2));
        vertex_down_1_screen(1, 0) = T(height) - (vertex_down_1_screen(1, 0) + T(1)) * (T(height) / T(2));
        vertex_down_2_screen(0, 0) = (vertex_down_2_screen(0, 0) + T(1)) * (T(width) / T(2));
        vertex_down_2_screen(1, 0) = T(height) - (vertex_down_2_screen(1, 0) + T(1)) * (T(height) / T(2));

        vertex_left_0_screen(0, 0) = (vertex_left_0_screen(0, 0) + T(1)) * (T(width) / T(2));
        vertex_left_0_screen(1, 0) = T(height) - (vertex_left_0_screen(1, 0) + T(1)) * (T(height) / T(2));
        vertex_left_1_screen(0, 0) = (vertex_left_1_screen(0, 0) + T(1)) * (T(width) / T(2));
        vertex_left_1_screen(1, 0) = T(height) - (vertex_left_1_screen(1, 0) + T(1)) * (T(height) / T(2));
        vertex_left_2_screen(0, 0) = (vertex_left_2_screen(0, 0) + T(1)) * (T(width) / T(2));
        vertex_left_2_screen(1, 0) = T(height) - (vertex_left_2_screen(1, 0) + T(1)) * (T(height) / T(2));

        vertex_right_0_screen(0, 0) = (vertex_right_0_screen(0, 0) + T(1)) * (T(width) / T(2));
        vertex_right_0_screen(1, 0) = T(height) - (vertex_right_0_screen(1, 0) + T(1)) * (T(height) / T(2));
        vertex_right_1_screen(0, 0) = (vertex_right_1_screen(0, 0) + T(1)) * (T(width) / T(2));
        vertex_right_1_screen(1, 0) = T(height) - (vertex_right_1_screen(1, 0) + T(1)) * (T(height) / T(2));
        vertex_right_2_screen(0, 0) = (vertex_right_2_screen(0, 0) + T(1)) * (T(width) / T(2));
        vertex_right_2_screen(1, 0) = T(height) - (vertex_right_2_screen(1, 0) + T(1)) * (T(height) / T(2));

        T one_over_v0ToLine12_top = T(1) / implicit_line(vertex_top_0_screen(0, 0), vertex_top_0_screen(1, 0), vertex_top_1_screen, vertex_top_2_screen);
        T one_over_v1ToLine20_top = T(1) / implicit_line(vertex_top_1_screen(0, 0), vertex_top_1_screen(1, 0), vertex_top_2_screen, vertex_top_0_screen);
        T one_over_v2ToLine01_top = T(1) / implicit_line(vertex_top_2_screen(0, 0), vertex_top_2_screen(1, 0), vertex_top_0_screen, vertex_top_1_screen);

        T one_over_v0ToLine12_down = T(1) / implicit_line(vertex_down_0_screen(0, 0), vertex_down_0_screen(1, 0), vertex_down_1_screen, vertex_down_2_screen);
        T one_over_v1ToLine20_down = T(1) / implicit_line(vertex_down_1_screen(0, 0), vertex_down_1_screen(1, 0), vertex_down_2_screen, vertex_down_0_screen);
        T one_over_v2ToLine01_down = T(1) / implicit_line(vertex_down_2_screen(0, 0), vertex_down_2_screen(1, 0), vertex_down_0_screen, vertex_down_1_screen);

        T one_over_v0ToLine12_left = T(1) / implicit_line(vertex_left_0_screen(0, 0), vertex_left_0_screen(1, 0), vertex_left_1_screen, vertex_left_2_screen);
        T one_over_v1ToLine20_left = T(1) / implicit_line(vertex_left_1_screen(0, 0), vertex_left_1_screen(1, 0), vertex_left_2_screen, vertex_left_0_screen);
        T one_over_v2ToLine01_left = T(1) / implicit_line(vertex_left_2_screen(0, 0), vertex_left_2_screen(1, 0), vertex_left_0_screen, vertex_left_1_screen);

        T one_over_v0ToLine12_right = T(1) / implicit_line(vertex_right_0_screen(0, 0), vertex_right_0_screen(1, 0), vertex_right_1_screen, vertex_right_2_screen);
        T one_over_v1ToLine20_right = T(1) / implicit_line(vertex_right_1_screen(0, 0), vertex_right_1_screen(1, 0), vertex_right_2_screen, vertex_right_0_screen);
        T one_over_v2ToLine01_right = T(1) / implicit_line(vertex_right_2_screen(0, 0), vertex_right_2_screen(1, 0), vertex_right_0_screen, vertex_right_1_screen);

        T alpha_bary_top = implicit_line(T(pixel_col), T(pixel_row - 1), vertex_top_1_screen, vertex_top_2_screen) * one_over_v0ToLine12_top;
        T beta_bary_top = implicit_line(T(pixel_col), T(pixel_row - 1), vertex_top_2_screen, vertex_top_0_screen) * one_over_v1ToLine20_top;
        T gamma_bary_top = implicit_line(T(pixel_col), T(pixel_row - 1), vertex_top_0_screen, vertex_top_1_screen) * one_over_v2ToLine01_top;

        T alpha_bary_down = implicit_line(T(pixel_col), T(pixel_row + 1), vertex_down_1_screen, vertex_down_2_screen) * one_over_v0ToLine12_down;
        T beta_bary_down = implicit_line(T(pixel_col), T(pixel_row + 1), vertex_down_2_screen, vertex_down_0_screen) * one_over_v1ToLine20_down;
        T gamma_bary_down = implicit_line(T(pixel_col), T(pixel_row + 1), vertex_down_0_screen, vertex_down_1_screen) * one_over_v2ToLine01_down;

        T alpha_bary_left = implicit_line(T(pixel_col - 1), T(pixel_row), vertex_left_1_screen, vertex_left_2_screen) * one_over_v0ToLine12_left;
        T beta_bary_left = implicit_line(T(pixel_col - 1), T(pixel_row), vertex_left_2_screen, vertex_left_0_screen) * one_over_v1ToLine20_left;
        T gamma_bary_left = implicit_line(T(pixel_col - 1), T(pixel_row), vertex_left_0_screen, vertex_left_1_screen) * one_over_v2ToLine01_left;

        T alpha_bary_right = implicit_line(T(pixel_col + 1), T(pixel_row), vertex_right_1_screen, vertex_right_2_screen) * one_over_v0ToLine12_right;
        T beta_bary_right = implicit_line(T(pixel_col + 1), T(pixel_row), vertex_right_2_screen, vertex_right_0_screen) * one_over_v1ToLine20_right;
        T gamma_bary_right = implicit_line(T(pixel_col + 1), T(pixel_row), vertex_right_0_screen, vertex_right_1_screen) * one_over_v2ToLine01_right;

        T vertex_x_top = alpha_bary_top * vertex_top_0_screen(0, 0) + beta_bary_top * vertex_top_1_screen(0, 0) + gamma_bary_top * vertex_top_2_screen(0, 0);
        T vertex_y_top = alpha_bary_top * vertex_top_0_screen(1, 0) + beta_bary_top * vertex_top_1_screen(1, 0) + gamma_bary_top * vertex_top_2_screen(1, 0);
        T vertex_z_top = alpha_bary_top * vertex_top_0_screen(2, 0) + beta_bary_top * vertex_top_1_screen(2, 0) + gamma_bary_top * vertex_top_2_screen(2, 0);

        T vertex_x_down = alpha_bary_down * vertex_down_0_screen(0, 0) + beta_bary_down * vertex_down_1_screen(0, 0) + gamma_bary_down * vertex_down_2_screen(0, 0);
        T vertex_y_down = alpha_bary_down * vertex_down_0_screen(1, 0) + beta_bary_down * vertex_down_1_screen(1, 0) + gamma_bary_down * vertex_down_2_screen(1, 0);
        T vertex_z_down = alpha_bary_down * vertex_down_0_screen(2, 0) + beta_bary_down * vertex_down_1_screen(2, 0) + gamma_bary_down * vertex_down_2_screen(2, 0);

        T vertex_x_left = alpha_bary_left * vertex_left_0_screen(0, 0) + beta_bary_left * vertex_left_1_screen(0, 0) + gamma_bary_left * vertex_left_2_screen(0, 0);
        T vertex_y_left = alpha_bary_left * vertex_left_0_screen(1, 0) + beta_bary_left * vertex_left_1_screen(1, 0) + gamma_bary_left * vertex_left_2_screen(1, 0);
        T vertex_z_left = alpha_bary_left * vertex_left_0_screen(2, 0) + beta_bary_left * vertex_left_1_screen(2, 0) + gamma_bary_left * vertex_left_2_screen(2, 0);

        T vertex_x_right = alpha_bary_right * vertex_right_0_screen(0, 0) + beta_bary_right * vertex_right_1_screen(0, 0) + gamma_bary_right * vertex_right_2_screen(0, 0);
        T vertex_y_right = alpha_bary_right * vertex_right_0_screen(1, 0) + beta_bary_right * vertex_right_1_screen(1, 0) + gamma_bary_right * vertex_right_2_screen(1, 0);
        T vertex_z_right = alpha_bary_right * vertex_right_0_screen(2, 0) + beta_bary_right * vertex_right_1_screen(2, 0) + gamma_bary_right * vertex_right_2_screen(2, 0);

        vertex_z_top = T(1) - ((T(2) * T(z_near) * T(z_far) / (T(z_far) + T(z_near) - vertex_z_top * (T(z_far) - T(z_near)))) - T(z_near)) / (T(z_far) - T(z_near));
        vertex_z_down = T(1) - ((T(2) * T(z_near) * T(z_far) / (T(z_far) + T(z_near) - vertex_z_down * (T(z_far) - T(z_near)))) - T(z_near)) / (T(z_far) - T(z_near));
        vertex_z_left = T(1) - ((T(2) * T(z_near) * T(z_far) / (T(z_far) + T(z_near) - vertex_z_left * (T(z_far) - T(z_near)))) - T(z_near)) / (T(z_far) - T(z_near));
        vertex_z_right = T(1) - ((T(2) * T(z_near) * T(z_far) / (T(z_far) + T(z_near) - vertex_z_right * (T(z_far) - T(z_near)))) - T(z_near)) / (T(z_far) - T(z_near));

        Vector<T, 3> dzdx;
        dzdx << vertex_x_right - vertex_x_left, vertex_y_right - vertex_y_left, vertex_z_right - vertex_z_left;
        Vector<T, 3> dzdy;
        dzdy << vertex_x_down - vertex_x_top, vertex_y_down - vertex_y_top, vertex_z_down - vertex_z_top;
        Vector<T, 3> point_normal_estimated = -(dzdx).cross(dzdy);
        point_normal_estimated.normalize();

        //      Point to plane distance from model to input
        residuals[1] = T(planeWeight) * ((vertex0_z - T(depth_source)) * T(point_normal_source(2)));

         //    Point to plane distance from input to model
        residuals[2] = T(planeWeight) * ((T(depth_source) - vertex0_z) * point_normal_estimated(2));
        return true;
    }
private:
    const double pointWeight, planeWeight, depth_source, pixel_row, pixel_col, width, height, z_near, z_far;
    const Vector3i vertex_indices_current_point, vertex_indices_top, vertex_indices_down, vertex_indices_left, vertex_indices_right;
    const Vector<int, 12> neighbour_vertex_indices;
    const Vector3d point_normal_source;
    const FaceModel* faceModel;
    const Matrix4d perspective_matrix;
};
//--------------------------------------------------Energy Terms Used for Face Reconstruction From Single Image--------------------------------------------------//


//--------------------------------------------------Energy Terms Designed for Facial Expression pipeline--------------------------------------------------//
class ExpressionFeatureSimilarityEnergy {
public:
    ExpressionFeatureSimilarityEnergy(const double& _landmarkWeight, const Vector2d& _landmark, const Face* _face, const FaceModel* _faceModel, const unsigned& _vertexIdx, Matrix4d _perspective_matrix,
        const unsigned& _viewport_width, const unsigned& _viewport_height) :
        landmarkWeight(_landmarkWeight),
        landmark(_landmark),
        face(_face),
        faceModel(_faceModel),
        vertexIdx(_vertexIdx),
        perspective_matrix(_perspective_matrix),
        viewport_width(_viewport_width),
        viewport_height(_viewport_height)
    { }

    template <typename T>
    bool operator()(const T const* gamma, const T* const extrinsicsArr, T* residuals) const {
        // ************* USING EIGEN (very slow...) ***************
        Map<const Matrix<T, -1, 1> > gammaMap(alpha, BFM_ALPHA_SIZE);
        // calculate vertex
        Matrix<T, -1, -1> vertex = (*face).getShapeBlock(3 * vertexIdx, 3).cast<T>()
            + (*faceModel).getExpBasisRowBlock(3 * vertexIdx, 3) * gammaMap;

        // apply pose (extrinsics)
        T vertex_transformed[3];
        PoseIncrement<T> pose_inc(const_cast<T* const>(extrinsicsArr));
        pose_inc.apply(vertex.data(), vertex_transformed);
        // apply perspective projection (instrinsics)
        Matrix<T, 4, 1> vertex_homogeneous;
        vertex_homogeneous << vertex_transformed[0], vertex_transformed[1], vertex_transformed[2], T(1);
        Matrix<T, -1, -1> vertex_clip_space = perspective_matrix * vertex_homogeneous;
        T x_pixel_space = (vertex_clip_space(0, 0) / vertex_clip_space(3, 0) + T(1)) * (T(viewport_width) / T(2));
        T y_pixel_space = T(viewport_height) - (vertex_clip_space(1, 0) / vertex_clip_space(3, 0) + T(1)) * (T(viewport_height) / T(2));
        residuals[0] = (T(landmark(0)) - x_pixel_space) * T(landmarkWeight);
        residuals[1] = (T(landmark(1)) - y_pixel_space) * T(landmarkWeight);
        return true;
    }

private:
    const double landmarkWeight;
    const Face* face;
    const FaceModel* faceModel;
    const Vector2d landmark;
    const unsigned vertexIdx, viewport_width, viewport_height;
    Matrix4d perspective_matrix;
};

class ExpressionGeometryPoint2PointConsistencyEnergy {
public:
    ExpressionGeometryPoint2PointConsistencyEnergy(const double& _pointWeight, const Vector3i& _vertex_indices, const Face* _face, const FaceModel* _faceModel,
        const double& _depth_source, Matrix4d _perspective_matrix, const double& _pixel_row, const double& _pixel_col,
        const double& _width, const double& _height) :
        pointWeight(_pointWeight),
        vertex_indices(_vertex_indices),
        faceModel(_faceModel),
        face(_face),
        depth_source(_depth_source),
        perspective_matrix(_perspective_matrix),
        pixel_row(_pixel_row),
        pixel_col(_pixel_col),
        width(_width),
        height(_height)
    { }

    template <typename T>
    bool operator() (const T const* gamma, const T* const extrinsicsArr, T* residuals) const {
        Map<const Matrix<T, -1, 1> > alphaMap(alpha, BFM_ALPHA_SIZE);
        Map<const Matrix<T, -1, 1> > gammaMap(gamma, BFM_GAMMA_SIZE);

        // calculate vertex
        Matrix<T, -1, -1> vertex_0 = (*faceModel).getShapeBlock(3 * vertex_indices(0), 3).cast<T>()
            + (*faceModel).getExpBasisRowBlock(3 * vertex_indices(0), 3) * gammaMap;

        Matrix<T, -1, -1> vertex_1 = (*faceModel).getShapeBlock(3 * vertex_indices(1), 3).cast<T>()
            + (*faceModel).getExpBasisRowBlock(3 * vertex_indices(1), 3) * gammaMap;

        Matrix<T, -1, -1> vertex_2 = (*faceModel).getShapeBlock(3 * vertex_indices(2), 3).cast<T>()
            + (*faceModel).getExpBasisRowBlock(3 * vertex_indices(2), 3) * gammaMap;

        // apply pose (extrinsics)
        T vertex_0_transformed[3];
        T vertex_1_transformed[3];
        T vertex_2_transformed[3];

        PoseIncrement<T> pose_inc(const_cast<T* const>(extrinsicsArr));
        pose_inc.apply(vertex_0.data(), vertex_0_transformed);
        pose_inc.apply(vertex_1.data(), vertex_1_transformed);
        pose_inc.apply(vertex_2.data(), vertex_2_transformed);

        Matrix<T, 4, 1> vertex_0_homogeneous;
        vertex_0_homogeneous << vertex_0_transformed[0], vertex_0_transformed[1], vertex_0_transformed[2], T(1);
        Matrix<T, 4, 1> vertex_1_homogeneous;
        vertex_1_homogeneous << vertex_1_transformed[0], vertex_1_transformed[1], vertex_1_transformed[2], T(1);
        Matrix<T, 4, 1> vertex_2_homogeneous;
        vertex_2_homogeneous << vertex_2_transformed[0], vertex_2_transformed[1], vertex_2_transformed[2], T(1);

        // apply perspective projection (instrinsics)
        Matrix<T, 4, 1> vertex_0_clip = perspective_matrix * vertex_0_homogeneous;
        Matrix<T, 4, 1> vertex_1_clip = perspective_matrix * vertex_1_homogeneous;
        Matrix<T, 4, 1> vertex_2_clip = perspective_matrix * vertex_2_homogeneous;

        T one_over_z0 = T(1) / vertex_0_clip(3, 0);
        T one_over_z1 = T(1) / vertex_1_clip(3, 0);
        T one_over_z2 = T(1) / vertex_2_clip(3, 0);

        Matrix<T, 4, 1> vertex_0_screen = vertex_0_clip / vertex_0_clip(3, 0);
        Matrix<T, 4, 1> vertex_1_screen = vertex_1_clip / vertex_1_clip(3, 0);
        Matrix<T, 4, 1> vertex_2_screen = vertex_2_clip / vertex_2_clip(3, 0);

        // To pixel space
        vertex_0_screen(0, 0) = (vertex_0_screen(0, 0) + T(1)) * (T(width) / T(2));
        vertex_0_screen(1, 0) = T(height) - (vertex_0_screen(1, 0) + T(1)) * (T(height) / T(2));
        vertex_1_screen(0, 0) = (vertex_1_screen(0, 0) + T(1)) * (T(width) / T(2));
        vertex_1_screen(1, 0) = T(height) - (vertex_1_screen(1, 0) + T(1)) * (T(height) / T(2));
        vertex_2_screen(0, 0) = (vertex_2_screen(0, 0) + T(1)) * (T(width) / T(2));
        vertex_2_screen(1, 0) = T(height) - (vertex_2_screen(1, 0) + T(1)) * (T(height) / T(2));

        T one_over_v0ToLine12 = T(1) / implicit_line(vertex_0_screen(0, 0), vertex_0_screen(1, 0), vertex_1_screen, vertex_2_screen);
        T one_over_v1ToLine20 = T(1) / implicit_line(vertex_1_screen(0, 0), vertex_1_screen(1, 0), vertex_2_screen, vertex_0_screen);
        T one_over_v2ToLine01 = T(1) / implicit_line(vertex_2_screen(0, 0), vertex_2_screen(1, 0), vertex_0_screen, vertex_1_screen);

        T alpha_bary = implicit_line(T(pixel_col), T(pixel_row), vertex_1_screen, vertex_2_screen) * one_over_v0ToLine12;
        T beta_bary = implicit_line(T(pixel_col), T(pixel_row), vertex_2_screen, vertex_0_screen) * one_over_v1ToLine20;
        T gamma_bary = implicit_line(T(pixel_col), T(pixel_row), vertex_0_screen, vertex_1_screen) * one_over_v2ToLine01;

        T x = alpha_bary * vertex_0_screen(0, 0) + beta_bary * vertex_1_screen(0, 0) + gamma_bary * vertex_2_screen(0, 0);
        T y = alpha_bary * vertex_0_screen(1, 0) + beta_bary * vertex_1_screen(1, 0) + gamma_bary * vertex_2_screen(1, 0);
        T depth = alpha_bary * vertex_0_screen(2, 0) + beta_bary * vertex_1_screen(2, 0) + gamma_bary * vertex_2_screen(2, 0);
        residuals[0] = (x - T(pixel_col)) * T(pointWeight);
        residuals[1] = (y - T(pixel_row)) * T(pointWeight);
        residuals[2] = (depth_source - depth) * T(pointWeight);
        return true;
    }
private:
    const double pointWeight, depth_source, pixel_row, pixel_col, width, height;
    const Vector3i vertex_indices;
    const Face* face;
    const FaceModel* faceModel;
    const Matrix4d perspective_matrix;
};

class ShCoefficientsConsistencyEnergy {
public:
    ShCoefficientsConsistencyEnergy(const double& weight, const Vector3d& source_color, const Vector3i& vertexIndices, const Vector3d& vertexWeights, const Face* face,
        const Vector<double, 9>& sh_basis_vertex_0, const Vector<double, 9>& sh_basis_vertex_1, const Vector<double, 9>& sh_basis_vertex_2) :
        //initialization
        m_weight{ weight },
        m_source_color{ source_color },
        m_vertexIndices{ vertexIndices },
        m_vertexWeights{ vertexWeights },
        m_face{ face },
        m_sh_basis_vertex_0{ sh_basis_vertex_0 },
        m_sh_basis_vertex_1{ sh_basis_vertex_1 },
        m_sh_basis_vertex_2{ sh_basis_vertex_2 }
    { }

    template <typename T>
    bool operator()(const T* const sh_red_coefficients, const T* const sh_green_coefficients, const T* const sh_blue_coefficients, T* residuals) const {
        Map<const Matrix<T, -1, 1> > shRedCoefficientsMap(sh_red_coefficients, 9);
        Map<const Matrix<T, -1, 1> > shGreenCoefficientsMap(sh_green_coefficients, 9);
        Map<const Matrix<T, -1, 1> > shBlueCoefficientsMap(sh_blue_coefficients, 9);

        Matrix<T, -1, -1> vertex_0_color = (*m_faceModel).getColorBlock(3 * m_vertexIndices(0), 3).cast<T>();
        Matrix<T, -1, -1> vertex_1_color = (*m_faceModel).getColorBlock(3 * m_vertexIndices(1), 3).cast<T>();
        Matrix<T, -1, -1> vertex_2_color = (*m_faceModel).getColorBlock(3 * m_vertexIndices(2), 3).cast<T>();

        T pi = T(3.1415926);

        T irradiance_vertex_0[3] = { T(0), T(0), T(0) };
        T irradiance_vertex_1[3] = { T(0), T(0), T(0) };
        T irradiance_vertex_2[3] = { T(0), T(0), T(0) };

        for (int j = 0; j < 9; ++j) {
            irradiance_vertex_0[0] += T(m_sh_basis_vertex_0(j)) * shRedCoefficientsMap(j, 0);
            irradiance_vertex_1[0] += T(m_sh_basis_vertex_1(j)) * shRedCoefficientsMap(j, 0);
            irradiance_vertex_2[0] += T(m_sh_basis_vertex_2(j)) * shRedCoefficientsMap(j, 0);
        }
        for (int j = 0; j < 9; ++j) {
            irradiance_vertex_0[1] += T(m_sh_basis_vertex_0(j)) * shGreenCoefficientsMap(j, 0);
            irradiance_vertex_1[1] += T(m_sh_basis_vertex_1(j)) * shGreenCoefficientsMap(j, 0);
            irradiance_vertex_2[1] += T(m_sh_basis_vertex_2(j)) * shGreenCoefficientsMap(j, 0);
        }
        for (int j = 0; j < 9; ++j) {
            irradiance_vertex_0[2] += T(m_sh_basis_vertex_0(j)) * shBlueCoefficientsMap(j, 0);
            irradiance_vertex_1[2] += T(m_sh_basis_vertex_1(j)) * shBlueCoefficientsMap(j, 0);
            irradiance_vertex_2[2] += T(m_sh_basis_vertex_2(j)) * shBlueCoefficientsMap(j, 0);
        }

        for (int i = 0; i < 3; ++i) {
            vertex_0_color(i) *= irradiance_vertex_0[i] / pi;
            vertex_1_color(i) *= irradiance_vertex_1[i] / pi;
            vertex_2_color(i) *= irradiance_vertex_2[i] / pi;
            residuals[i] = (T(m_source_color[i]) / T(255) - ((vertex_0_color(i) * T(m_vertexWeights(0)) + vertex_1_color(i) * T(m_vertexWeights(1)) + vertex_2_color(i) * T(m_vertexWeights(2))))) * m_weight;
        }
        return true;
    }
private:
    const Vector3d m_source_color;
    const Vector3i m_vertexIndices;
    const Vector3d m_vertexWeights;
    const Vector<double, 9> m_sh_basis_vertex_0, m_sh_basis_vertex_1, m_sh_basis_vertex_2;
    const Face* m_face;
    const double m_weight;
};
//--------------------------------------------------Energy Terms Designed for Facial Expression pipeline--------------------------------------------------//


//-----------------------------------------------------------Energy Terms Used In Both Cases-------------------------------------------------------------//
class ColorConsistencyEnergy {
public:
    ColorConsistencyEnergy(const double& weight, const Vector3d& source_color, const Vector3i& vertexIndices, const Vector3d& vertexWeights, const FaceModel* faceModel,
        const Vector<double, 9>& sh_basis_vertex_0, const Vector<double, 9>& sh_basis_vertex_1, const Vector<double, 9>& sh_basis_vertex_2) :
        //initialization
        m_weight{ weight },
        m_source_color{ source_color },
        m_vertexIndices{ vertexIndices },
        m_vertexWeights{ vertexWeights },
        m_faceModel{ faceModel },
        m_sh_basis_vertex_0{ sh_basis_vertex_0 },
        m_sh_basis_vertex_1{ sh_basis_vertex_1 },
        m_sh_basis_vertex_2{ sh_basis_vertex_2 }
    { }

    template <typename T>
    bool operator()(const T* const beta, const T* const sh_red_coefficients, const T* const sh_green_coefficients, const T* const sh_blue_coefficients, T* residuals) const {
        Map<const Matrix<T, -1, 1> > betaMap(beta, BFM_BETA_SIZE);
        Map<const Matrix<T, -1, 1> > shRedCoefficientsMap(sh_red_coefficients, 9);
        Map<const Matrix<T, -1, 1> > shGreenCoefficientsMap(sh_green_coefficients, 9);
        Map<const Matrix<T, -1, 1> > shBlueCoefficientsMap(sh_blue_coefficients, 9);

        Matrix<T, -1, -1> vertex_0_color = (*m_faceModel).getColorMeanBlock(3 * m_vertexIndices(0), 3).cast<T>()
            + (*m_faceModel).getColorBasisRowBlock(3 * m_vertexIndices(0), 3) * betaMap;
        Matrix<T, -1, -1> vertex_1_color = (*m_faceModel).getColorMeanBlock(3 * m_vertexIndices(1), 3).cast<T>()
            + (*m_faceModel).getColorBasisRowBlock(3 * m_vertexIndices(1), 3) * betaMap;
        Matrix<T, -1, -1> vertex_2_color = (*m_faceModel).getColorMeanBlock(3 * m_vertexIndices(2), 3).cast<T>()
            + (*m_faceModel).getColorBasisRowBlock(3 * m_vertexIndices(2), 3) * betaMap;

        T pi = T(3.1415926);

        T irradiance_vertex_0[3] = { T(0), T(0), T(0) };
        T irradiance_vertex_1[3] = { T(0), T(0), T(0) };
        T irradiance_vertex_2[3] = { T(0), T(0), T(0) };

        for (int j = 0; j < 9; ++j) {
            irradiance_vertex_0[0] += T(m_sh_basis_vertex_0(j)) * shRedCoefficientsMap(j, 0);
            irradiance_vertex_1[0] += T(m_sh_basis_vertex_1(j)) * shRedCoefficientsMap(j, 0);
            irradiance_vertex_2[0] += T(m_sh_basis_vertex_2(j)) * shRedCoefficientsMap(j, 0);
        }
        for (int j = 0; j < 9; ++j) {
            irradiance_vertex_0[1] += T(m_sh_basis_vertex_0(j)) * shGreenCoefficientsMap(j, 0);
            irradiance_vertex_1[1] += T(m_sh_basis_vertex_1(j)) * shGreenCoefficientsMap(j, 0);
            irradiance_vertex_2[1] += T(m_sh_basis_vertex_2(j)) * shGreenCoefficientsMap(j, 0);
        }
        for (int j = 0; j < 9; ++j) {
            irradiance_vertex_0[2] += T(m_sh_basis_vertex_0(j)) * shBlueCoefficientsMap(j, 0);
            irradiance_vertex_1[2] += T(m_sh_basis_vertex_1(j)) * shBlueCoefficientsMap(j, 0);
            irradiance_vertex_2[2] += T(m_sh_basis_vertex_2(j)) * shBlueCoefficientsMap(j, 0);
        }

        for (int i = 0; i < 3; ++i) {
            vertex_0_color(i) *= irradiance_vertex_0[i] / pi;
            vertex_1_color(i) *= irradiance_vertex_1[i] / pi;
            vertex_2_color(i) *= irradiance_vertex_2[i] / pi;
            residuals[i] = (T(m_source_color[i])/T(255) - ((vertex_0_color(i) * T(m_vertexWeights(0)) + vertex_1_color(i) * T(m_vertexWeights(1)) + vertex_2_color(i) * T(m_vertexWeights(2))))) * m_weight;
        }
        return true;
    }
private:
    const Vector3d m_source_color;
    const Vector3i m_vertexIndices;
    const Vector3d m_vertexWeights;
    const Vector<double, 9> m_sh_basis_vertex_0, m_sh_basis_vertex_1, m_sh_basis_vertex_2;
    const FaceModel* m_faceModel;
    const double m_weight;
};

class TextureColorConsistencyEnergy {
public:
    TextureColorConsistencyEnergy(const double& weight, const Vector3d& source_color, const Vector3d& vertexWeights, const FaceModel* faceModel,
        const Vector<double, 9>& sh_basis_vertex_0, const Vector<double, 9>& sh_basis_vertex_1, const Vector<double, 9>& sh_basis_vertex_2,
        const Vector<double, 9>& sh_red_coefficients, const Vector<double, 9>& sh_green_coefficients, const Vector<double, 9>& sh_blue_coefficients) :
        //initialization
        m_weight{ weight },
        m_source_color{ source_color },
        m_vertexWeights{ vertexWeights },
        m_faceModel{ faceModel },
        m_sh_basis_vertex_0{ sh_basis_vertex_0 },
        m_sh_basis_vertex_1{ sh_basis_vertex_1 },
        m_sh_basis_vertex_2{ sh_basis_vertex_2 },
        m_sh_red_coefficients{ sh_red_coefficients },
        m_sh_green_coefficients{ sh_green_coefficients },
        m_sh_blue_coefficients{ sh_blue_coefficients }
    { }

    template <typename T>
    bool operator()(const T* const vertex_color_0, const T* const vertex_color_1, const T* const vertex_color_2, T* residuals) const {
        T pi = T(3.1415926);

        T irradiance_vertex_0[3] = { T(0), T(0), T(0) };
        T irradiance_vertex_1[3] = { T(0), T(0), T(0) };
        T irradiance_vertex_2[3] = { T(0), T(0), T(0) };

        for (int j = 0; j < 9; ++j) {
            irradiance_vertex_0[0] += T(m_sh_basis_vertex_0(j)) * m_sh_red_coefficients(j, 0);
            irradiance_vertex_1[0] += T(m_sh_basis_vertex_1(j)) * m_sh_red_coefficients(j, 0);
            irradiance_vertex_2[0] += T(m_sh_basis_vertex_2(j)) * m_sh_red_coefficients(j, 0);
        }

        for (int j = 0; j < 9; ++j) {
            irradiance_vertex_0[1] += T(m_sh_basis_vertex_0(j)) * m_sh_green_coefficients(j, 0);
            irradiance_vertex_1[1] += T(m_sh_basis_vertex_1(j)) * m_sh_green_coefficients(j, 0);
            irradiance_vertex_2[1] += T(m_sh_basis_vertex_2(j)) * m_sh_green_coefficients(j, 0);
        }

        for (int j = 0; j < 9; ++j) {
            irradiance_vertex_0[2] += T(m_sh_basis_vertex_0(j)) * m_sh_blue_coefficients(j, 0);
            irradiance_vertex_1[2] += T(m_sh_basis_vertex_1(j)) * m_sh_blue_coefficients(j, 0);
            irradiance_vertex_2[2] += T(m_sh_basis_vertex_2(j)) * m_sh_blue_coefficients(j, 0);
        }

        T vertex_color_0_[3] = { T(0), T(0), T(0) };
        T vertex_color_1_[3] = { T(0), T(0), T(0) };
        T vertex_color_2_[3] = { T(0), T(0), T(0) };

        for (int i = 0; i < 3; ++i) {
            vertex_color_0_[i] = vertex_color_0[i] * irradiance_vertex_0[i] / pi;
            vertex_color_1_[i] = vertex_color_1[i] * irradiance_vertex_1[i] / pi;
            vertex_color_2_[i] = vertex_color_2[i] * irradiance_vertex_2[i] / pi;

            vertex_color_0_[i] = max(vertex_color_0_[i], T(0));
            vertex_color_1_[i] = max(vertex_color_1_[i], T(0));
            vertex_color_2_[i] = max(vertex_color_2_[i], T(0));

            residuals[i] = (T(m_source_color(i)) - (vertex_color_0_[i] * T(m_vertexWeights(0)) + vertex_color_1_[i] * T(m_vertexWeights(1)) + vertex_color_2_[i] * T(m_vertexWeights(2))) * T(255.)) * m_weight;
        }
        return true;
    }
private:
    const Vector3d m_source_color;
    const Vector3d m_vertexWeights;
    const Vector<double, 9> m_sh_basis_vertex_0, m_sh_basis_vertex_1, m_sh_basis_vertex_2, m_sh_red_coefficients, m_sh_green_coefficients, m_sh_blue_coefficients;
    const FaceModel* m_faceModel;
    const double m_weight;
};

class RegularizationEnergy {
public:
    RegularizationEnergy(const double _weight, const unsigned& _size) : weight(_weight), size(_size) {}
    template <typename T>
    bool operator()(const T const* terms, T* residuals) const {
        for (unsigned i = 0; i < size; i++)
            residuals[i] = T(sqrt(weight)) * terms[i];
        return true;
    }
private:
    const double weight;
    const unsigned size;
};
//-----------------------------------------------------------Energy Terms Used In Both Cases-------------------------------------------------------------//



class Optimizer {

public:
    // maybe add some options later when we are done with the basic version. Like GN/LM, Cuda/Cpu....
    Optimizer(double _landmakrWeight = 0.125, double _pointWeight = 0.141, double _planeWeight = 0.316, double _colorWeight = 4.47, double _shapeRegWeight = 0.05, double _expRegWeight = 0.05, double _coloRegWeight = 0.05, int _maxIteration = 20) {
        landmarkWeight = _landmakrWeight;
        pointWeight = _pointWeight;
        planeWeight = _planeWeight;
        colorWeight = _colorWeight;
        shapeRegWeight = _shapeRegWeight;
        expRegWeight = _expRegWeight;
        colorRegWeight = _coloRegWeight;
        maxIteration = _maxIteration;
    }

    void optimize(Face& face, int option) {
        Renderer render = Renderer::Get();
        Image img = face.getImage();
        render.initialiaze_rendering_context(face.getFaceModel(), img.getHeight(), img.getWidth());

        // Extrinsic setting
        double poseArr[6];
        PoseIncrement<double>::extrinsicsMatTo6DoG(face.getExtrinsics(), poseArr);
        PoseIncrement poseIncrement = PoseIncrement<double>(poseArr);

        // Face parameter weights
        FaceModel faceModel = face.getFaceModel();
        VectorXd alpha = face.getAlpha();
        VectorXd beta = face.getBeta();
        VectorXd gamma = face.getGamma();
        VectorXd shape = face.getShape();
        VectorXd color = face.getColor();
        VectorXd sh_red_coefficients = face.getSHRedCoefficients();
        VectorXd sh_green_coefficients = face.getSHGreenCoefficients();
        VectorXd sh_blue_coefficients = face.getSHBlueCoefficients();

        unsigned numLandmarks = face.getFaceModel().getNumLandmarks();

        MatrixXd source_depth = img.getDepthMap();
        cv::Mat point_normal = img.getNormalMap();
        vector<MatrixXd> source_color = img.getRGB();

        if (option == 0) {
            reconstruct_face(face, render, img, poseIncrement, faceModel, alpha, beta, gamma, shape, color, sh_red_coefficients, sh_green_coefficients,
                sh_blue_coefficients, numLandmarks, source_depth, source_color, point_normal);
        }
        else {
            return;
        }
        
    }
private:
    //--------------------------------------------------Functions for face reconstruction--------------------------------------------------//
    void reconstruct_face(Face& face, Renderer& render, Image& img, PoseIncrement<double>& poseIncrement, FaceModel& faceModel, VectorXd& alpha, VectorXd& beta,
        VectorXd& gamma, VectorXd& shape, VectorXd& color, VectorXd& sh_red_coefficients, VectorXd& sh_green_coefficients, VectorXd& sh_blue_coefficients, unsigned numLandmarks,
        MatrixXd& source_depth, vector<MatrixXd>& source_color, cv::Mat& point_normal) {

        // Estimate shape, expression and illumination
        for (int i = 0; i < maxIteration; ++i) {
            // Create ceres problem
            ceres::Problem problem;
            // Add energy terms
            add_landmark_terms(problem, numLandmarks, img, faceModel, face, alpha, gamma, poseIncrement);
            add_regularization_terms(problem, alpha, beta, gamma);
            int intern_iteration = 20;
            if (i > 0) {
                add_geometry_and_color_terms(problem, render, face, img, faceModel, source_depth, source_color, alpha, gamma, beta, poseIncrement, sh_red_coefficients,
                    sh_green_coefficients, sh_blue_coefficients, point_normal);
                intern_iteration = 1;
            }

            // Solve problem
            solve(problem, intern_iteration, ceres::SPARSE_NORMAL_CHOLESKY);
            
            // UPDATE PARAMS
            face.setAlpha(alpha);
            face.setGamma(gamma);
            face.setBeta(beta);
            face.setExtrinsics(PoseIncrement<double>::convertToMatrix(poseIncrement));
            face.setSHRedCoefficients(sh_red_coefficients);
            face.setSHGreenCoefficients(sh_green_coefficients);
            face.setSHBlueCoefficients(sh_blue_coefficients);
        }

        face.setShape(face.calculateVerticesDefault());
        face.setColor(face.calculateColorsDefault());

        // Estimate texture from estimated shape, expression, color and illumination
        VectorXd estimated_color = face.getColor();
        ceres::Problem problem;
        add_texture_terms(problem, render, face, img, faceModel, source_depth, source_color, estimated_color);
        solve(problem, 20, ceres::ITERATIVE_SCHUR);
        face.setColor(estimated_color);
    }

    void add_landmark_terms(ceres::Problem& problem, int numLandmarks, Image& img, FaceModel& faceModel, Face& face, VectorXd& alpha, VectorXd& gamma, PoseIncrement<double>& poseIncrement) {
        for (unsigned i = 0; i < numLandmarks; i++)
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<FeatureSimilarityEnergy, 2, BFM_ALPHA_SIZE, BFM_GAMMA_SIZE, 6>
                (new FeatureSimilarityEnergy(landmarkWeight, img.getLandmark(i), &faceModel, faceModel.getLandmarkVertexIdx(i), face.getIntrinsics(), img.getWidth(), img.getHeight())),
                nullptr, alpha.data(), gamma.data(), poseIncrement.getData()
            );
    }

    void add_regularization_terms(ceres::Problem& problem, VectorXd& alpha, VectorXd& beta, VectorXd& gamma) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<RegularizationEnergy, BFM_ALPHA_SIZE, BFM_ALPHA_SIZE>
            (new RegularizationEnergy(shapeRegWeight, BFM_ALPHA_SIZE)),
            nullptr, alpha.data()
        );
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<RegularizationEnergy, BFM_BETA_SIZE, BFM_BETA_SIZE>
            (new RegularizationEnergy(colorRegWeight, BFM_BETA_SIZE)),
            nullptr, beta.data()
        );
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<RegularizationEnergy, BFM_GAMMA_SIZE, BFM_GAMMA_SIZE>
            (new RegularizationEnergy(expRegWeight, BFM_GAMMA_SIZE)),
            nullptr, gamma.data()
        );
    }

    void add_geometry_and_color_terms(ceres::Problem& problem, Renderer& render, Face& face, Image& img, FaceModel& faceModel, MatrixXd& source_depth,
        vector<MatrixXd>& source_color, VectorXd& alpha, VectorXd& gamma, VectorXd& beta, PoseIncrement<double>& poseIncrement, VectorXd& sh_red_coefficients,
        VectorXd& sh_green_coefficients, VectorXd& sh_blue_coefficients, cv::Mat& point_normal) {

        render.clear_buffers();

        Matrix4f mvp_matrix = face.getFullProjectionMatrix().cast<float>().transpose();
        Matrix4f mv_matrix = face.getExtrinsics().cast<float>().transpose();
        Matrix4d projection_matrix = face.getIntrinsics();
        VectorXf vertices = face.calculateVerticesDefault().cast<float>();
        VectorXf colors = face.calculateColorsDefault().cast<float>();
        VectorXf sh_red_coefficients_ = face.getSHRedCoefficients().cast<float>();
        VectorXf sh_green_coefficients_ = face.getSHGreenCoefficients().cast<float>();
        VectorXf sh_blue_coefficients_ = face.getSHBlueCoefficients().cast<float>();

        render.render(mvp_matrix, mv_matrix, vertices, colors, sh_red_coefficients_, sh_green_coefficients_, sh_blue_coefficients_, face.get_z_near(), face.get_z_far());
        cv::Mat color_buffer = render.get_color_buffer();
        cv::Mat indices_buffer = render.get_pixel_triangle_buffer();
        cv::Mat bary_coords = render.get_pixel_bary_coord_buffer();
        cv::Mat pixel_triangle_normal_buffer = render.get_pixel_triangle_normal_buffer();
        cv::Mat rendered_depth_buffer = render.get_depth_buffer();
        
        cv::imshow("depth", rendered_depth_buffer);
        cv::imshow("color", color_buffer);
        cv::waitKey(0);

        for (unsigned i = 0; i < img.getHeight(); ++i) {
            for (unsigned j = 0; j < img.getWidth(); ++j) {
                float depth = rendered_depth_buffer.at<float>(i, j);
                if (depth != 0 && source_depth(i, j) != 0) {
                    Vector3i indices = faceModel.getTriangulationByRow(indices_buffer.at<int>(i, j));
                    Vector3d bary_coords_;
                    bary_coords_ << bary_coords.at<cv::Vec3d>(i, j)[0], bary_coords.at<cv::Vec3d>(i, j)[1], bary_coords.at<cv::Vec3d>(i, j)[2];

                    if ((i > 0) && (i < img.getHeight() - 1) && (j > 0) && (j < img.getWidth() - 1)) {
                        float depth_top = rendered_depth_buffer.at<float>(i - 1, j);
                        float depth_down = rendered_depth_buffer.at <float> (i + 1, j);
                        float depth_left = rendered_depth_buffer.at<float>(i, j - 1);
                        float depth_right = rendered_depth_buffer.at<float>(i, j + 1);

                        if (depth_top != 0 && depth_down != 0 && depth_left != 0 && depth_right != 0 &&
                            source_depth(i - 1, j) != 0 && source_depth(i + 1, j) != 0 && source_depth(i, j - 1) != 0 && source_depth(i, j + 1) != 0) {

                            Vector3i indices_top = faceModel.getTriangulationByRow(indices_buffer.at<int>(i - 1, j));
                            Vector3i indices_down = faceModel.getTriangulationByRow(indices_buffer.at<int>(i + 1, j));
                            Vector3i indices_left = faceModel.getTriangulationByRow(indices_buffer.at<int>(i, j - 1));
                            Vector3i indices_right = faceModel.getTriangulationByRow(indices_buffer.at<int>(i, j + 1));

                            Vector3d point_normal_source;
                            point_normal_source << point_normal.at<cv::Vec3d>(i, j)[0], point_normal.at<cv::Vec3d>(i, j)[1], point_normal.at<cv::Vec3d>(i, j)[2];

                            // Add geometry term
                            problem.AddResidualBlock(
                                new ceres::AutoDiffCostFunction<GeometryConsistencyEnergy, 3, BFM_ALPHA_SIZE, BFM_GAMMA_SIZE, 6>
                                (new GeometryConsistencyEnergy(pointWeight, planeWeight, indices, indices_top, indices_down, indices_left, indices_right,
                                    source_depth(i, j) / 255., point_normal_source, &faceModel, projection_matrix, double(j) + 0.5, double(i) + 0.5,
                                    double(img.getWidth()), double(img.getHeight()), face.get_z_near(), face.get_z_far())),
                                nullptr, alpha.data(), gamma.data(), poseIncrement.getData()
                            );
                        }

                    }

                    // Add color term
                    Vector3d source_color_;
                    source_color_ << double(source_color[0](i, j)), double(source_color[1](i, j)), double(source_color[2](i, j));

                    cv::Vec<float, 9> triangle_vertex_normals = pixel_triangle_normal_buffer.at<cv::Vec<float, 9>>(i, j);
                    Vector3d normal_vertex_0, normal_vertex_1, normal_vertex_2;
                    normal_vertex_0 << double(triangle_vertex_normals[0]), double(triangle_vertex_normals[1]), double(triangle_vertex_normals[2]);
                    normal_vertex_1 << double(triangle_vertex_normals[3]), double(triangle_vertex_normals[4]), double(triangle_vertex_normals[5]);
                    normal_vertex_2 << double(triangle_vertex_normals[6]), double(triangle_vertex_normals[7]), double(triangle_vertex_normals[8]);

                    normal_vertex_0.normalize();
                    normal_vertex_1.normalize();
                    normal_vertex_2.normalize();

                    Vector<double, 9> sh_basis_vertex_0, sh_basis_vertex_1, sh_basis_vertex_2;

                    for (int i = 0; i < 9; ++i) {
                        sh_basis_vertex_0(i) = SH_basis_function(normal_vertex_0, i);
                        sh_basis_vertex_1(i) = SH_basis_function(normal_vertex_1, i);
                        sh_basis_vertex_2(i) = SH_basis_function(normal_vertex_2, i);
                    }

                    problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<ColorConsistencyEnergy, 3, BFM_BETA_SIZE, 9, 9, 9>
                        (new ColorConsistencyEnergy(colorWeight, source_color_, indices, bary_coords_, &faceModel, sh_basis_vertex_0, sh_basis_vertex_1, sh_basis_vertex_2)),
                        nullptr, beta.data(), sh_red_coefficients.data(), sh_green_coefficients.data(), sh_blue_coefficients.data()
                    );
                }
            }
        }
    }

    void add_texture_terms(ceres::Problem& problem, Renderer& render, Face& face, Image& img, FaceModel& faceModel, MatrixXd& source_depth,
        vector<MatrixXd>& source_color, VectorXd& color) {

        render.clear_buffers();

        Matrix4f mvp_matrix = face.getFullProjectionMatrix().cast<float>().transpose();
        Matrix4f mv_matrix = face.getExtrinsics().cast<float>().transpose();
        VectorXf vertices = face.calculateVerticesDefault().cast<float>();
        VectorXf colors = face.calculateColorsDefault().cast<float>();
        VectorXf sh_red_coefficients_ = face.getSHRedCoefficients().cast<float>();
        VectorXf sh_green_coefficients_ = face.getSHGreenCoefficients().cast<float>();
        VectorXf sh_blue_coefficients_ = face.getSHBlueCoefficients().cast<float>();

        render.render(mvp_matrix, mv_matrix, vertices, colors, sh_red_coefficients_, sh_green_coefficients_, sh_blue_coefficients_, face.get_z_near(), face.get_z_far());

        cv::Mat indices_buffer = render.get_pixel_triangle_buffer();
        cv::Mat bary_coords = render.get_pixel_bary_coord_buffer();
        cv::Mat pixel_triangle_normal_buffer = render.get_pixel_triangle_normal_buffer();
        cv::Mat depth_buffer = render.get_depth_buffer();
        for (unsigned i = 0; i < img.getHeight(); ++i) {
            for (unsigned j = 0; j < img.getWidth(); ++j) {
                float depth = depth_buffer.at<float>(i, j);
                if (!(depth == 0)) {
                    // Add texture color term
                    Vector3i indices = faceModel.getTriangulationByRow(indices_buffer.at<int>(i, j));

                    Vector3d bary_coords_;
                    bary_coords_ << bary_coords.at<cv::Vec3d>(i, j)[0], bary_coords.at<cv::Vec3d>(i, j)[1], bary_coords.at<cv::Vec3d>(i, j)[2];

                    Vector3d source_color_;
                    source_color_ << double(source_color[0](i, j)), double(source_color[1](i, j)), double(source_color[2](i, j));

                    cv::Vec<float, 9> triangle_vertex_normals = pixel_triangle_normal_buffer.at<cv::Vec<float, 9>>(i, j);
                    Vector3d normal_vertex_0, normal_vertex_1, normal_vertex_2;
                    normal_vertex_0 << double(triangle_vertex_normals[0]), double(triangle_vertex_normals[1]), double(triangle_vertex_normals[2]);
                    normal_vertex_1 << double(triangle_vertex_normals[3]), double(triangle_vertex_normals[4]), double(triangle_vertex_normals[5]);
                    normal_vertex_2 << double(triangle_vertex_normals[6]), double(triangle_vertex_normals[7]), double(triangle_vertex_normals[8]);

                    normal_vertex_0.normalize();
                    normal_vertex_1.normalize();
                    normal_vertex_2.normalize();

                    Vector<double, 9> sh_basis_vertex_0, sh_basis_vertex_1, sh_basis_vertex_2;

                    for (int i = 0; i < 9; ++i) {
                        sh_basis_vertex_0(i) = SH_basis_function(normal_vertex_0, i);
                        sh_basis_vertex_1(i) = SH_basis_function(normal_vertex_1, i);
                        sh_basis_vertex_2(i) = SH_basis_function(normal_vertex_2, i);
                    }

                    problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<TextureColorConsistencyEnergy, 3, 3, 3, 3>
                        (new TextureColorConsistencyEnergy(colorWeight, source_color_, bary_coords_, &faceModel, sh_basis_vertex_0, sh_basis_vertex_1, sh_basis_vertex_2,
                            face.getSHRedCoefficients(), face.getSHGreenCoefficients(), face.getSHBlueCoefficients())),
                        nullptr, &color.data()[indices(0) * 3], &color.data()[indices(1) * 3], &color.data()[indices(2) * 3]
                    );
                }
            }
        }
    }
    //--------------------------------------------------Functions for face reconstruction--------------------------------------------------//


    // --------------------------------------------------General functions--------------------------------------------------//
    void solve(ceres::Problem& problem, int max_iterations, ceres::LinearSolverType solver_type) {
        ceres::Solver::Options options;
        //options.dense_linear_algebra_library_type = ceres::CUDA;
        options.num_threads = omp_get_max_threads();
        options.max_num_iterations = max_iterations;
        options.linear_solver_type = solver_type;
        // options.preconditioner_type = ceres::JACOBI;
        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << std::endl;
    }
    // --------------------------------------------------General functions--------------------------------------------------//

    double landmarkWeight, shapeRegWeight, expRegWeight, colorRegWeight, pointWeight, planeWeight, colorWeight;
    int maxIteration;
};