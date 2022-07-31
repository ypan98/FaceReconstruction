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
* Feature Similarity (Landmarks) energy optimization. Landmarks of the image are precalculated with a deep learning model in python and stored in a file, and BFM17 
* landmarks are given by the model.
* This energy term aims to aliniate the landmark points
*/
class FeatureSimilarityEnergy {
public:
    FeatureSimilarityEnergy(const double& _landmarkWeight, const Vector2d& _landmark, const double& _depth, const FaceModel* _faceModel, const Face* _face,
        const unsigned& _vertexIdx, Matrix4d _perspective_matrix, const unsigned& _viewport_width, const unsigned& _viewport_height, const double& _z_near, 
        const double& _z_far, const bool& _expression) :
        landmarkWeight(_landmarkWeight),
        landmark(_landmark),
        depth(_depth),
        faceModel(_faceModel),
        face(_face),
        vertexIdx(_vertexIdx),
        perspective_matrix(_perspective_matrix),
        viewport_width(_viewport_width),
        viewport_height(_viewport_height),
        z_near(_z_near),
        z_far(_z_far),
        expression(_expression)
    { }

    template <typename T>
    bool operator()(const T const* alpha, const T const* gamma, const T* const extrinsicsArr, T* residuals) const {
        Map<const Matrix<T, -1, 1> > alphaMap(alpha, BFM_ALPHA_SIZE);
        Map<const Matrix<T, -1, 1> > gammaMap(gamma, BFM_GAMMA_SIZE);
        // calculate vertex
        Matrix<T, -1, -1> vertex;
        if (!expression) {
            vertex = (*faceModel).getShapeMeanBlock(3 * vertexIdx, 3).cast<T>()
                + (*faceModel).getExpMeanBlock(3 * vertexIdx, 3).cast<T>()
                + (*faceModel).getShapeBasisRowBlock(3 * vertexIdx, 3) * alphaMap
                + (*faceModel).getExpBasisRowBlock(3 * vertexIdx, 3) * gammaMap;
        }
        else {
            vertex = (*face).getShapeBlock(3 * vertexIdx, 3).cast<T>()
                + (*faceModel).getExpBasisRowBlock(3 * vertexIdx, 3) * gammaMap;
        }

        // apply pose (extrinsics)
        T vertex_transformed[3];
        PoseIncrement<T> pose_inc(const_cast<T* const>(extrinsicsArr));
        pose_inc.apply(vertex.data(), vertex_transformed);
        // apply perspective projection (instrinsics)
        Matrix<T, 4, 1> vertex_homogeneous;
        vertex_homogeneous << vertex_transformed[0], vertex_transformed[1], vertex_transformed[2], T(1);
        Matrix<T, -1, -1> vertex_clip_space = perspective_matrix * vertex_homogeneous;
        vertex_clip_space /= vertex_clip_space(3, 0);
        T x_pixel_space = (vertex_clip_space(0, 0) + T(1)) * (T(viewport_width) / T(2));
        T y_pixel_space = T(viewport_height) - (vertex_clip_space(1, 0) + T(1)) * (T(viewport_height) / T(2));
        T z = T(1) - ((T(2) * T(z_near) * T(z_far) / (T(z_far) + T(z_near) - vertex_clip_space(2, 0) * (T(z_far) - T(z_near)))) - T(z_near)) / (T(z_far) - T(z_near));
        residuals[0] = (T(landmark(0)) - x_pixel_space)*T(landmarkWeight);
        residuals[1] = (T(landmark(1)) - y_pixel_space)*T(landmarkWeight);
        residuals[2] = (T(depth) - z) * T(landmarkWeight);
        return true;
    }

private:
    const double landmarkWeight, depth, z_near, z_far;
    const FaceModel* faceModel;
    const Face* face;
    const Vector2d landmark;
    const unsigned vertexIdx, viewport_width, viewport_height;
    Matrix4d perspective_matrix;
    const bool expression;
};
/*
* Geometry point 2 point energy optimization. Tries to minimize the L2 distance between the projected vertices (face and baricentric coordinate calculated from rasterization step)
* with the points from the image's depth map.
* If expression = true, then only gamma (expression) is optimized.
*/  
class GeometryPoint2PointConsistencyEnergy {
public:
    GeometryPoint2PointConsistencyEnergy(const double& _pointWeight, const Vector3i& _vertex_indices, const Vector3d& _bary_coord, const double& _depth_source, 
        const FaceModel* _faceModel, const Face* _face, const Matrix4d& _perspective_matrix, const double& _pixel_row, const double& _pixel_col, const double& _width, 
        const double& _height, const double& _z_near, const double& _z_far, const bool& _expression) :
        pointWeight(_pointWeight),
        vertex_indices(_vertex_indices),
        bary_coord(_bary_coord),
        depth_source(_depth_source),
        faceModel(_faceModel),
        face(_face),
        perspective_matrix(_perspective_matrix),
        pixel_row(_pixel_row),
        pixel_col(_pixel_col),
        width(_width),
        height(_height),
        z_near(_z_near),
        z_far(_z_far),
        expression(_expression)
    { }

    template <typename T>
    bool operator() (const T const* alpha, const T const* gamma, const T* const extrinsicsArr, T* residuals) const {
        Map<const Matrix<T, BFM_ALPHA_SIZE, 1> > alphaMap(alpha);
        Map<const Matrix<T, BFM_GAMMA_SIZE, 1> > gammaMap(gamma);

        Matrix<T, 4, 3> vertices;
        PoseIncrement<T> pose_inc(const_cast<T* const>(extrinsicsArr));
        for (int i = 0; i < 3; ++i) {
            // Calculate vertices
            if (!expression) {
                vertices.block(0, i, 3, 1) = (*faceModel).getShapeMeanBlock(3 * vertex_indices(i), 3).cast<T>()
                    + (*faceModel).getExpMeanBlock(3 * vertex_indices(i), 3).cast<T>()
                    + (*faceModel).getShapeBasisRowBlock(3 * vertex_indices(i), 3) * alphaMap
                    + (*faceModel).getExpBasisRowBlock(3 * vertex_indices(i), 3) * gammaMap;
            }
            else {
                vertices.block(0, i, 3, 1) = (*face).getShapeBlock(3 * vertex_indices(i), 3).cast<T>()
                    + (*faceModel).getExpBasisRowBlock(3 * vertex_indices(i), 3) * gammaMap;
            }
            vertices(3, i) = T(1);

            // Apply transformation
            T* data;
            data = &vertices.data()[i * 4];
            pose_inc.apply(data, data);

            // Apply perspective projection
            vertices.col(i) = perspective_matrix * vertices.col(i);

            // To clip space
            vertices.col(i) = vertices.col(i) / vertices.col(i)(3);

            // To pixel space
            vertices(0, i) = (vertices(0, i) + T(1)) * (T(width) / T(2));
            vertices(1, i) = T(height) - (vertices(1, i) + T(1)) * (T(height) / T(2));
        }

        T vertex_coord[3];
        for (int i = 0; i < 3; ++i) {
            vertex_coord[i] = bary_coord(0) * vertices.col(0)(i) + bary_coord(1) * vertices.col(1)(i) + bary_coord(2) * vertices.col(2)(i);
        }
        vertex_coord[2] = T(1) - ((T(2) * T(z_near) * T(z_far) / (T(z_far) + T(z_near) - vertex_coord[2] * (T(z_far) - T(z_near)))) - T(z_near)) / (T(z_far) - T(z_near));

        //      Point to point distance
        residuals[0] = (vertex_coord[0] - pixel_col) * T(pointWeight);
        residuals[1] = (vertex_coord[1] - pixel_row) * T(pointWeight);
        residuals[2] = (vertex_coord[2] - depth_source) * T(pointWeight);
        return true;
    }
private:
    const double pointWeight, depth_source, pixel_row, pixel_col, width, height, z_near, z_far;
    const Vector3i vertex_indices;
    const Vector3d point_normal_source, bary_coord;
    const FaceModel* faceModel;
    const Face* face;
    const Matrix4d perspective_matrix;
    const bool expression;
};
/*
* Geometry point 2 plane energy optimization. Tries to minimize the L2 distance between the projected vertices (face and baricentric coordinate calculated from rasterization step)
* with the tangent plane from the points of the image's depth map, and viceversa (this is why we have 2 residuals).
* If expression = true, then only gamma (expression) is optimized.
*/
class GeometryPoint2PlaneConsistencyEnergy {
public:
    GeometryPoint2PlaneConsistencyEnergy(const double& _planeWeight, const Vector<int, 15>& _vertex_indices, const Vector<double, 15>& _bary_coords, const double& _depth_source, 
        const Vector3d& _point_normal_source, const FaceModel* _faceModel, const Face* _face, const Matrix4d& _perspective_matrix, const double& _pixel_row, const double& _pixel_col, 
        const double& _width, const double& _height, const double& _z_near, const double& _z_far, const bool& _expression) :
        planeWeight(_planeWeight),
        vertex_indices(_vertex_indices),
        bary_coords(_bary_coords),
        point_normal_source(_point_normal_source),
        depth_source(_depth_source),
        faceModel(_faceModel),
        face(_face),
        perspective_matrix(_perspective_matrix),
        pixel_row(_pixel_row),
        pixel_col(_pixel_col),
        width(_width),
        height(_height),
        z_near(_z_near),
        z_far(_z_far),
        expression(_expression)
    { }

    template <typename T>
    bool operator() (const T const* alpha, const T const* gamma, const T* const extrinsicsArr, T* residuals) const {
        Map<const Matrix<T, BFM_ALPHA_SIZE, 1> > alphaMap(alpha);
        Map<const Matrix<T, BFM_GAMMA_SIZE, 1> > gammaMap(gamma);

        Matrix<T, 4, 15> vertices;
        PoseIncrement<T> pose_inc(const_cast<T* const>(extrinsicsArr));
        for (int i = 0; i < 15; ++i) {
            // Calculate vertices
            if (!expression) {
                vertices.block(0, i, 3, 1) = (*faceModel).getShapeMeanBlock(3 * vertex_indices(i), 3).cast<T>()
                    + (*faceModel).getExpMeanBlock(3 * vertex_indices(i), 3).cast<T>()
                    + (*faceModel).getShapeBasisRowBlock(3 * vertex_indices(i), 3) * alphaMap
                    + (*faceModel).getExpBasisRowBlock(3 * vertex_indices(i), 3) * gammaMap;
            }
            else {
                vertices.block(0, i, 3, 1) = (*face).getShapeBlock(3 * vertex_indices(i), 3).cast<T>()
                    + (*faceModel).getExpBasisRowBlock(3 * vertex_indices(i), 3) * gammaMap;
            }
            vertices(3, i) = T(1);

            // Apply transformation
            T* data;
            data = &vertices.data()[i * 4];
            pose_inc.apply(data, data);

            // Apply perspective projection
            vertices.col(i) = perspective_matrix * vertices.col(i);

            // To clip space
            vertices.col(i) = vertices.col(i) / vertices.col(i)(3);

            // To pixel space
            vertices(0, i) = (vertices(0, i) + T(1)) * (T(width) / T(2));
            vertices(1, i) = T(height) - (vertices(1, i) + T(1)) * (T(height) / T(2));
        }

        T vertex_coords[15];
        for (int i = 0; i < 5; ++i) {
            vertex_coords[i * 3] = T(bary_coords(i * 3)) * vertices.col(i * 3)(0) + T(bary_coords(i * 3 + 1)) * vertices.col(i * 3 + 1)(0) + T(bary_coords(i * 3 + 2)) * vertices.col(i * 3 + 2)(0);
            vertex_coords[i * 3 + 1] = T(bary_coords(i * 3)) * vertices.col(i * 3)(1) + T(bary_coords(i * 3 + 1)) * vertices.col(i * 3 + 1)(1) + T(bary_coords(i * 3 + 2)) * vertices.col(i * 3 + 2)(1);
            vertex_coords[i * 3 + 2] = T(bary_coords(i * 3)) * vertices.col(i * 3)(2) + T(bary_coords(i * 3 + 1)) * vertices.col(i * 3 + 1)(2) + T(bary_coords(i * 3 + 2)) * vertices.col(i * 3 + 2)(2);

        }
        for (int i = 0; i < 5; ++i) {
            vertex_coords[i * 3 + 2] = T(1) - ((T(2) * T(z_near) * T(z_far) / (T(z_far) + T(z_near) - vertex_coords[i * 3 + 2] * (T(z_far) - T(z_near)))) - T(z_near)) / (T(z_far) - T(z_near));
        }

        Vector<T, 3> dzdx;
        dzdx << vertex_coords[12] - vertex_coords[9], vertex_coords[10]-vertex_coords[13], vertex_coords[14] - vertex_coords[11];
        Vector<T, 3> dzdy;
        dzdy << vertex_coords[6] - vertex_coords[3], vertex_coords[7] - vertex_coords[4], vertex_coords[8] - vertex_coords[5];
        Vector<T, 3> point_normal_estimated = -(dzdx).cross(dzdy);
        point_normal_estimated.normalize();

        //      Point to plane distance from model to input
        residuals[0] = T(planeWeight) * ((vertex_coords[0] - T(pixel_col)) * T(point_normal_source(0)) +
            (vertex_coords[1] - T(pixel_row)) * T(point_normal_source(1)) +
            (vertex_coords[2] - T(depth_source)) * T(point_normal_source(2)));
        //      Point to plane distance from input to model
        residuals[1] = T(planeWeight) * ((T(pixel_col) - vertex_coords[0]) * point_normal_estimated(0) +
            (T(pixel_row) - vertex_coords[1]) * point_normal_estimated(1) +
            (T(depth_source) - vertex_coords[2]) * point_normal_estimated(2));
        return true;
    }
private:
    const double planeWeight, depth_source, pixel_row, pixel_col, width, height, z_near, z_far;
    const Vector<int, 15> vertex_indices;
    const Vector<double, 15> bary_coords;
    const Vector3d point_normal_source;
    const FaceModel* faceModel;
    const Face* face;
    const Matrix4d perspective_matrix;
    const bool expression;
};
/*
* Color consistency energy optimization. Tries to minimize the difference between the projected vertices' color (face and baricentric coordinate calculated from rasterization step)
* with the colors from the input rgb image. Illumination from spherical harmonics coefficients are taken into account.
* If expression = true, then only illumination (spherical harmonics parameters) are optimized. As changing expression doesnt change base color.
*/
class ColorConsistencyEnergy {
public:
    ColorConsistencyEnergy(const double& weight, const Vector3d& source_color, const Vector3i& vertexIndices, const Vector3d& vertexWeights, const FaceModel* faceModel,
        const Face* face, const Vector<double, 9>& sh_basis_vertex_0, const Vector<double, 9>& sh_basis_vertex_1, const Vector<double, 9>& sh_basis_vertex_2, const bool& expression) :
        //initialization
        m_weight{ weight },
        m_source_color{ source_color },
        m_vertexIndices{ vertexIndices },
        m_vertexWeights{ vertexWeights },
        m_faceModel{ faceModel },
        m_face{ face },
        m_sh_basis_vertex_0{ sh_basis_vertex_0 },
        m_sh_basis_vertex_1{ sh_basis_vertex_1 },
        m_sh_basis_vertex_2{ sh_basis_vertex_2 },
        m_expression{ expression }
    { }

    template <typename T>
    bool operator()(const T* const beta, const T* const sh_red_coefficients, const T* const sh_green_coefficients, const T* const sh_blue_coefficients, T* residuals) const {
        Map<const Matrix<T, -1, 1> > betaMap(beta, BFM_BETA_SIZE);
        Map<const Matrix<T, -1, 1> > shRedCoefficientsMap(sh_red_coefficients, 9);
        Map<const Matrix<T, -1, 1> > shGreenCoefficientsMap(sh_green_coefficients, 9);
        Map<const Matrix<T, -1, 1> > shBlueCoefficientsMap(sh_blue_coefficients, 9);

        Matrix<T, -1, -1> vertex_0_color;
        Matrix<T, -1, -1> vertex_1_color;
        Matrix<T, -1, -1> vertex_2_color;

        if (!m_expression) {
            vertex_0_color = (*m_faceModel).getColorMeanBlock(3 * m_vertexIndices(0), 3).cast<T>()
                + (*m_faceModel).getColorBasisRowBlock(3 * m_vertexIndices(0), 3) * betaMap;
            vertex_1_color = (*m_faceModel).getColorMeanBlock(3 * m_vertexIndices(1), 3).cast<T>()
                + (*m_faceModel).getColorBasisRowBlock(3 * m_vertexIndices(1), 3) * betaMap;
            vertex_2_color = (*m_faceModel).getColorMeanBlock(3 * m_vertexIndices(2), 3).cast<T>()
                + (*m_faceModel).getColorBasisRowBlock(3 * m_vertexIndices(2), 3) * betaMap;
        }
        else {
            vertex_0_color = (*m_face).getColorBlock(3 * m_vertexIndices(0), 3).cast<T>();
            vertex_1_color = (*m_face).getColorBlock(3 * m_vertexIndices(1), 3).cast<T>();
            vertex_2_color = (*m_face).getColorBlock(3 * m_vertexIndices(2), 3).cast<T>();
        }

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

            vertex_0_color(i) = min(max(vertex_0_color(i), T(0)), T(1));
            vertex_1_color(i) = min(max(vertex_1_color(i), T(0)), T(1));
            vertex_2_color(i) = min(max(vertex_2_color(i), T(0)), T(1));

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
    const Face* m_face;
    const double m_weight;
    const bool m_expression;
};
/*
* Parameter regularization energy term. Tries to keep the parameters (coefficients) color to the prior (mean). That is closing the values to zero
*/
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
    // constructor with optimizer options and energy term weights
    Optimizer(Face& _sourceFace, bool _downsample = true, double _landmakrWeight = 0.35, double _pointWeight = 1.14, double _planeWeight = 3.34, double _colorWeight = 4.47, 
        double _shapeRegWeight = 0.05, double _expRegWeight = 0.05, double _coloRegWeight = 0.05, int _maxIteration = 3):sourceFace(_sourceFace) {
        downsample = _downsample;

        // Initialize energy weights
        landmarkWeight = _landmakrWeight;
        pointWeight = _pointWeight;
        planeWeight = _planeWeight;
        colorWeight = _colorWeight;
        shapeRegWeight = _shapeRegWeight;
        expRegWeight = _expRegWeight;
        colorRegWeight = _coloRegWeight;
        maxIteration = _maxIteration;

        // Initialize renderers
        Image img = sourceFace.getImage();
        rendererOriginal = Renderer(sourceFace.getFaceModel(), img.getHeight(), img.getWidth());
        rendererDownsampled = Renderer(sourceFace.getFaceModel(), img.getHeightDown(), img.getWidthDown());
    }

    // optimize for the face parameters
    void optimize(bool estimate_expression_only = false, bool show_intermediate_results = true) {
        Image img = sourceFace.getImage();
        // Extrinsic setting
        double poseArr[6];
        PoseIncrement<double>::extrinsicsMatTo6DoG(sourceFace.getExtrinsics(), poseArr);
        PoseIncrement poseIncrement = PoseIncrement<double>(poseArr);
        // Face parameter weights
        FaceModel faceModel = sourceFace.getFaceModel();
        VectorXd alpha = sourceFace.getAlpha();
        VectorXd beta = sourceFace.getBeta();
        VectorXd gamma = sourceFace.getGamma();
        VectorXd shape = sourceFace.getShape();
        VectorXd color = sourceFace.getColor();
        VectorXd sh_red_coefficients = sourceFace.getSHRedCoefficients();
        VectorXd sh_green_coefficients = sourceFace.getSHGreenCoefficients();
        VectorXd sh_blue_coefficients = sourceFace.getSHBlueCoefficients();
        unsigned numLandmarks = sourceFace.getFaceModel().getNumLandmarks();
        MatrixXd source_depth = img.getDepthMap();
        MatrixXd source_depth_down = img.getDepthMapDown();
        cv::Mat source_point_normal = img.getNormalMap();
        cv::Mat source_point_normal_down = img.getNormalMapDown();
        vector<MatrixXd> source_color = img.getRGB();
        vector<MatrixXd> source_color_down = img.getRGBDown();

        // reconstruct face or only expression and illumination parameters depending on option
        fit_face(img, poseIncrement, faceModel, alpha, beta, gamma, shape, color, sh_red_coefficients, sh_green_coefficients,
            sh_blue_coefficients, numLandmarks, source_depth, source_depth_down, source_color, source_color_down, source_point_normal, 
            source_point_normal_down, estimate_expression_only, show_intermediate_results);
        
    }
private:
    //--------------------------------------------------Functions for face reconstruction--------------------------------------------------//

    void fit_face(Image& img, PoseIncrement<double>& poseIncrement, FaceModel& faceModel, VectorXd& alpha, VectorXd& beta,
        VectorXd& gamma, VectorXd& shape, VectorXd& color, VectorXd& sh_red_coefficients, VectorXd& sh_green_coefficients, VectorXd& sh_blue_coefficients, 
        unsigned numLandmarks, MatrixXd& source_depth, MatrixXd& source_depth_down, vector<MatrixXd>& source_color, vector<MatrixXd>& source_color_down,
        cv::Mat& source_point_normal, cv::Mat& source_point_normal_down, bool estimate_expression_only, bool show_intermediate_results) {
        // Estimate shape, expression and illumination
        for (int i = 0; i < maxIteration; ++i) {
            // Create ceres problem
            ceres::Problem problem;
            int intern_iteration = 20;
            // Add energy terms
            if (downsample) {
                add_landmark_terms(problem, numLandmarks, img, source_depth_down, faceModel, alpha, gamma, poseIncrement, estimate_expression_only);
                add_regularization_terms(problem, alpha, beta, gamma);
                if (i > 0) {
                    add_geometry_and_color_terms(problem, img, faceModel, source_depth_down, source_color_down, alpha, gamma, beta, poseIncrement, sh_red_coefficients,
                        sh_green_coefficients, sh_blue_coefficients, source_point_normal_down, estimate_expression_only, show_intermediate_results, i);
                    intern_iteration = 1;
                }
            }
            else {
                add_landmark_terms(problem, numLandmarks, img, source_depth, faceModel, alpha, gamma, poseIncrement, estimate_expression_only);
                add_regularization_terms(problem, alpha, beta, gamma);
                if (i > 0) {
                    add_geometry_and_color_terms(problem, img, faceModel, source_depth, source_color, alpha, gamma, beta, poseIncrement, sh_red_coefficients,
                        sh_green_coefficients, sh_blue_coefficients, source_point_normal, estimate_expression_only, show_intermediate_results, i);
                    intern_iteration = 1;
                }
            }
            // Solve problem
            solve(problem, intern_iteration, ceres::ITERATIVE_SCHUR);
            // UPDATE PARAMS
            sourceFace.setAlpha(alpha);
            sourceFace.setGamma(gamma);
            sourceFace.setBeta(beta);
            sourceFace.setExtrinsics(PoseIncrement<double>::convertToMatrix(poseIncrement));
            sourceFace.setSHRedCoefficients(sh_red_coefficients);
            sourceFace.setSHGreenCoefficients(sh_green_coefficients);
            sourceFace.setSHBlueCoefficients(sh_blue_coefficients);
        }
        // set base color and shape (neutral expression) 
        if (!estimate_expression_only) {
            sourceFace.setShape(sourceFace.calculateVerticesNeutralExp());
            sourceFace.setColor(sourceFace.calculateColorsDefault());
            // Estimate texture from estimated shape, expression, color and illumination
            VectorXd estimated_color = sourceFace.getColor();
            //optimize_texture(img, faceModel, source_depth, source_color, estimated_color);
            sourceFace.setColor(estimated_color);
        }
    }

    // add landmakr terms to the problem
    void add_landmark_terms(ceres::Problem& problem, int numLandmarks, Image& img, MatrixXd& source_depth, FaceModel& faceModel,
        VectorXd& alpha, VectorXd& gamma, PoseIncrement<double>& poseIncrement, bool expression) {
        for (unsigned i = 0; i < numLandmarks; i++) {
            Vector2d landmark_coord;
            unsigned int width, height;

            if (downsample) {
                landmark_coord = img.getLandmark(i) / img.getDownScale();
                width = img.getWidthDown();
                height = img.getHeightDown();
            }
            else {
                landmark_coord = img.getLandmark(i);
                width = img.getWidth();
                height = img.getHeight();
            }

            double depth = source_depth(int(landmark_coord(1)), int(landmark_coord(0))) / 255.;
            if (depth > 0) {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<FeatureSimilarityEnergy, 3, BFM_ALPHA_SIZE, BFM_GAMMA_SIZE, 6>
                    (new FeatureSimilarityEnergy(landmarkWeight, landmark_coord, depth, &faceModel, &sourceFace, faceModel.getLandmarkVertexIdx(i),
                        sourceFace.getIntrinsics(), width, height, sourceFace.get_z_near(), sourceFace.get_z_far(), expression)),
                    nullptr, alpha.data(), gamma.data(), poseIncrement.getData()
                );
            }
        }
    }

    // add regularization terms to the probelm
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

    // add geometry and color terms
    void add_geometry_and_color_terms(ceres::Problem& problem, Image& img, FaceModel& faceModel, MatrixXd& source_depth,
        vector<MatrixXd>& source_color, VectorXd& alpha, VectorXd& gamma, VectorXd& beta, PoseIncrement<double>& poseIncrement, 
        VectorXd& sh_red_coefficients, VectorXd& sh_green_coefficients, VectorXd& sh_blue_coefficients, 
        cv::Mat& point_normal, bool estimate_expression_only, bool show_intermediate_results, int iteration) {

        rendererOriginal.clear_buffers();
        rendererDownsampled.clear_buffers();

        Matrix4f mvp_matrix = sourceFace.getFullProjectionMatrix().cast<float>().transpose();
        Matrix4f mv_matrix = sourceFace.getExtrinsics().cast<float>().transpose();
        Matrix4d projection_matrix = sourceFace.getIntrinsics();
        VectorXf vertices = sourceFace.calculateVerticesDefault().cast<float>();
        VectorXf colors = sourceFace.calculateColorsDefault().cast<float>();
        VectorXf sh_red_coefficients_ = sourceFace.getSHRedCoefficients().cast<float>();
        VectorXf sh_green_coefficients_ = sourceFace.getSHGreenCoefficients().cast<float>();
        VectorXf sh_blue_coefficients_ = sourceFace.getSHBlueCoefficients().cast<float>();

        cv::Mat color_buffer, indices_buffer, bary_coords, pixel_triangle_normal_buffer, rendered_depth_buffer;
        unsigned int height, width;

        if (downsample) {
            rendererDownsampled.render(mvp_matrix, mv_matrix, vertices, colors, sh_red_coefficients_, sh_green_coefficients_, sh_blue_coefficients_, sourceFace.get_z_near(), sourceFace.get_z_far());
            color_buffer = rendererDownsampled.get_color_buffer();
            indices_buffer = rendererDownsampled.get_pixel_triangle_buffer();
            bary_coords = rendererDownsampled.get_pixel_bary_coord_buffer();
            pixel_triangle_normal_buffer = rendererDownsampled.get_pixel_triangle_normal_buffer();
            rendered_depth_buffer = rendererDownsampled.get_depth_buffer();
            height = img.getHeightDown();
            width = img.getWidthDown();
        }
        else {
            rendererOriginal.render(mvp_matrix, mv_matrix, vertices, colors, sh_red_coefficients_, sh_green_coefficients_, sh_blue_coefficients_, sourceFace.get_z_near(), sourceFace.get_z_far());
            color_buffer = rendererOriginal.get_color_buffer();
            indices_buffer = rendererOriginal.get_pixel_triangle_buffer();
            bary_coords = rendererOriginal.get_pixel_bary_coord_buffer();
            pixel_triangle_normal_buffer = rendererOriginal.get_pixel_triangle_normal_buffer();
            rendered_depth_buffer = rendererOriginal.get_depth_buffer();
            height = img.getHeight();
            width = img.getWidth();
        }

        if (show_intermediate_results) {
            cv::imshow("rendered face model at iteration: " + std::to_string(iteration), color_buffer);
            cv::waitKey(0);
        }

        for (unsigned i = 0; i < height; ++i) {
            for (unsigned j = 0; j < width; ++j) {
                float depth = rendered_depth_buffer.at<float>(i, j);
                if (depth != 0 && source_depth(i, j) != 0) {
                    Vector3i indices = faceModel.getTriangulationByRow(indices_buffer.at<int>(i, j));
                    Vector3d perspective_corrected_bary_coord;
                    perspective_corrected_bary_coord << bary_coords.at<cv::Vec6d>(i, j)[3], bary_coords.at<cv::Vec6d>(i, j)[4], bary_coords.at<cv::Vec6d>(i, j)[5];
                    if ((i > 0) && (i < img.getHeight() - 1) && (j > 0) && (j < img.getWidth() - 1)) {
                        float depth_top = rendered_depth_buffer.at<float>(i - 1, j);
                        float depth_down = rendered_depth_buffer.at <float> (i + 1, j);
                        float depth_left = rendered_depth_buffer.at<float>(i, j - 1);
                        float depth_right = rendered_depth_buffer.at<float>(i, j + 1);

                        if (depth_top != 0 && depth_down != 0 && depth_left != 0 && depth_right != 0 &&
                            source_depth(i - 1, j) != 0 && source_depth(i + 1, j) != 0 && source_depth(i, j - 1) != 0 && source_depth(i, j + 1) != 0) {

                            Vector<int, 15> vertex_indices;
                            Vector3i indices_top = faceModel.getTriangulationByRow(indices_buffer.at<int>(i - 1, j));
                            Vector3i indices_down = faceModel.getTriangulationByRow(indices_buffer.at<int>(i + 1, j));
                            Vector3i indices_left = faceModel.getTriangulationByRow(indices_buffer.at<int>(i, j - 1));
                            Vector3i indices_right = faceModel.getTriangulationByRow(indices_buffer.at<int>(i, j + 1));
                            vertex_indices << indices(0), indices(1), indices(2),
                                indices_top(0), indices_top(1), indices_top(2),
                                indices_down(0), indices_down(1), indices_down(2),
                                indices_left(0), indices_left(1), indices_left(2),
                                indices_right(0), indices_right(1), indices_right(2);

                            Vector<double, 15> vertices_affine_bary_coords;
                            vertices_affine_bary_coords << bary_coords.at<cv::Vec6d>(i, j)[0], bary_coords.at<cv::Vec6d>(i, j)[1], bary_coords.at<cv::Vec6d>(i, j)[2],
                                bary_coords.at<cv::Vec6d>(i - 1, j)[0], bary_coords.at<cv::Vec6d>(i - 1, j)[1], bary_coords.at<cv::Vec6d>(i - 1, j)[2],
                                bary_coords.at<cv::Vec6d>(i + 1, j)[0], bary_coords.at<cv::Vec6d>(i + 1, j)[1], bary_coords.at<cv::Vec6d>(i + 1, j)[2],
                                bary_coords.at<cv::Vec6d>(i, j - 1)[0], bary_coords.at<cv::Vec6d>(i, j - 1)[1], bary_coords.at<cv::Vec6d>(i, j - 1)[2],
                                bary_coords.at<cv::Vec6d>(i, j + 1)[0], bary_coords.at<cv::Vec6d>(i, j + 1)[1], bary_coords.at<cv::Vec6d>(i, j + 1)[2];
                            
                            Vector3d point_normal_source;
                            point_normal_source << point_normal.at<cv::Vec3d>(i, j)[0], point_normal.at<cv::Vec3d>(i, j)[1], point_normal.at<cv::Vec3d>(i, j)[2];

                            // Add p2plane term
                            problem.AddResidualBlock(
                                new ceres::AutoDiffCostFunction<GeometryPoint2PlaneConsistencyEnergy, 2, BFM_ALPHA_SIZE, BFM_GAMMA_SIZE, 6>
                                (new GeometryPoint2PlaneConsistencyEnergy(planeWeight, vertex_indices, vertices_affine_bary_coords,
                                    source_depth(i, j) / 255., point_normal_source, &faceModel, &sourceFace, projection_matrix, double(i) + 0.5, double(j) + 0.5,
                                    double(width), double(height), sourceFace.get_z_near(), sourceFace.get_z_far(), estimate_expression_only)),
                                nullptr, alpha.data(), gamma.data(), poseIncrement.getData()
                            );
                        }

                    }
                    // Add p2p term
                    problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<GeometryPoint2PointConsistencyEnergy, 3, BFM_ALPHA_SIZE, BFM_GAMMA_SIZE, 6>
                        (new GeometryPoint2PointConsistencyEnergy(pointWeight, indices, perspective_corrected_bary_coord,
                            source_depth(i, j) / 255., &faceModel, &sourceFace, projection_matrix, double(i) + 0.5, double(j) + 0.5,
                            double(width), double(height), sourceFace.get_z_near(), sourceFace.get_z_far(), estimate_expression_only)),
                        nullptr, alpha.data(), gamma.data(), poseIncrement.getData()
                    );

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
                        (new ColorConsistencyEnergy(colorWeight, source_color_, indices, perspective_corrected_bary_coord, &faceModel, &sourceFace, sh_basis_vertex_0,
                            sh_basis_vertex_1, sh_basis_vertex_2, estimate_expression_only)),
                        nullptr, beta.data(), sh_red_coefficients.data(), sh_green_coefficients.data(), sh_blue_coefficients.data()
                    );
                }
            }
        }
    }

    // optimize for texture (by backprojecting the colors to the correspoding vertices), without Ceres
    void optimize_texture(Image& img, FaceModel& faceModel, MatrixXd& source_depth,
        vector<MatrixXd>& source_color, VectorXd& color) {
        rendererOriginal.clear_buffers();

        Matrix4f mvp_matrix = sourceFace.getFullProjectionMatrix().cast<float>().transpose();
        Matrix4f mv_matrix = sourceFace.getExtrinsics().cast<float>().transpose();
        VectorXf vertices = sourceFace.getShape().cast<float>();
        VectorXf colors = sourceFace.getColor().cast<float>();
        VectorXf sh_red_coefficients_ = sourceFace.getSHRedCoefficients().cast<float>();
        VectorXf sh_green_coefficients_ = sourceFace.getSHGreenCoefficients().cast<float>();
        VectorXf sh_blue_coefficients_ = sourceFace.getSHBlueCoefficients().cast<float>();

        rendererOriginal.render(mvp_matrix, mv_matrix, vertices, colors, sh_red_coefficients_, sh_green_coefficients_, sh_blue_coefficients_, sourceFace.get_z_near(), sourceFace.get_z_far());

        cv::Mat indices_buffer = rendererOriginal.get_pixel_triangle_buffer();
        cv::Mat bary_coords = rendererOriginal.get_pixel_bary_coord_buffer();
        cv::Mat pixel_triangle_normal_buffer = rendererOriginal.get_pixel_triangle_normal_buffer();
        cv::Mat depth_buffer = rendererOriginal.get_depth_buffer();

        double pi = 3.1415926;
        int height = img.getHeight();
        int width = img.getWidth();
        mvp_matrix.transposeInPlace();
        for (unsigned i = 0; i < height; ++i) {
            for (unsigned j = 0; j < width; ++j) {
                float depth = depth_buffer.at<float>(i, j);
                if (!(depth == 0)) {
                    Vector3i indices = faceModel.getTriangulationByRow(indices_buffer.at<int>(i, j));

                    cv::Vec<float, 9> triangle_vertex_normals = pixel_triangle_normal_buffer.at<cv::Vec<float, 9>>(i, j);
                    for (int z = 0; z < 3; ++z) {
                        Vector4f vertex;
                        vertex.block(0, 0, 3, 1) = sourceFace.getShapeBlock(3 * indices(z), 3).cast<float>();
                        vertex(3) = 1.0;

                        // Apply perspective projection
                        vertex = mvp_matrix * vertex;

                        // To clip space
                        vertex = vertex / vertex(3);

                        // To pixel space
                        vertex(0) = (vertex(0) + 1.) * (width / 2.);
                        vertex(1) = height - (vertex(1) + 1.) * (height / 2);
                        Vector3d normal;
                        normal << double(triangle_vertex_normals[z * 3]), double(triangle_vertex_normals[z * 3 + 1]), double(triangle_vertex_normals[z * 3 + 2]);
                        normal.normalize();

                        Vector<float, 9> sh_basis;
                        for (int k = 0; k < 9; ++k) {
                            sh_basis(k) = SH_basis_function(normal, k);
                        }

                        double irradiance[3] = { 0, 0, 0 };

                        for (int k = 0; k < 9; ++k) {
                            irradiance[0] += sh_basis(k) * sh_red_coefficients_(k);
                            irradiance[1] += sh_basis(k) * sh_green_coefficients_(k);
                            irradiance[2] += sh_basis(k) * sh_blue_coefficients_(k);
                        }

                        color.data()[indices(z) * 3] = (source_color[0](int(vertex(1)), int(vertex(0))) * pi) / (irradiance[0] * 255.);
                        color.data()[indices(z) * 3 + 1] = (source_color[1](int(vertex(1)), int(vertex(0))) * pi) / (irradiance[1] * 255.);
                        color.data()[indices(z) * 3 + 2] = (source_color[2](int(vertex(1)), int(vertex(0))) * pi) / (irradiance[2] * 255.);
                    }
                }
            }
        }
    }
    //--------------------------------------------------Functions for face reconstruction--------------------------------------------------//

    // --------------------------------------------------General functions--------------------------------------------------//
    void solve(ceres::Problem& problem, int max_iterations, ceres::LinearSolverType solver_type) {
        ceres::Solver::Options options;
        options.dense_linear_algebra_library_type = ceres::CUDA;
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

    bool downsample;
    Face &sourceFace;
    Renderer rendererDownsampled, rendererOriginal;
    double landmarkWeight, shapeRegWeight, expRegWeight, colorRegWeight, pointWeight, planeWeight, colorWeight;
    int maxIteration;
};