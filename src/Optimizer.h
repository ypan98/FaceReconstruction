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
    double pi = 3.1415926;
    switch (basis_index)
    {
    case 0:
        return 1. / 2. * sqrt(1. / pi);
    case 1:
        return sqrt(3. / (4. * pi)) * normal(2);
    case 2:
        return sqrt(3. / (4. * pi)) * normal(1);
    case 3:
        return sqrt(3. / (4. * pi)) * normal(0);
    case 4:
        return 1. / 2. * sqrt(15. / pi) * normal(0) * normal(2);
    case 5:
        return 1. / 2. * sqrt(15. / pi) * normal(2) * normal(1);
    case 6:
        return 1. / 4. * sqrt(5. / pi) * (-normal(0) * normal(0) - normal(2) * normal(2) + 2 * normal(1) * normal(1));
    case 7:
        return 1. / 2. * sqrt(15. / pi) * normal(1) * normal(0);
    case 8:
        return 1. / 4. * sqrt(15. / pi) * (normal(0) * normal(0) - normal(2) * normal(2));
    default:
        return 0.;
    }
};

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
        // ************* USING EIGEN (very slow...) ***************
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

/*
* Regularization term
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

class GeometryPoint2PointConsistencyEnergy {
public:
    GeometryPoint2PointConsistencyEnergy(const double& _pointWeight, const Vector3i& _vertex_indices, const FaceModel* _faceModel,
        const double& _depth_source, Matrix4d _perspective_matrix, const double& _pixel_row, const double& _pixel_col, 
        const double& _width, const double& _height, const Vector3d& _barycentric_coords) :
        pointWeight(_pointWeight),
        vertex_indices(_vertex_indices),
        faceModel(_faceModel),
        depth_source(_depth_source),
        perspective_matrix(_perspective_matrix),
        pixel_row(_pixel_row),
        pixel_col(_pixel_col),
        width(_width),
        height(_height),
        barycentric_coords(_barycentric_coords)
    { }

    template <typename T>
    bool operator() (const T const* alpha, const T const* gamma, const T* const extrinsicsArr, T* residuals) const{
        Map<const Matrix<T, -1, 1> > alphaMap(alpha, BFM_ALPHA_SIZE);
        Map<const Matrix<T, -1, 1> > gammaMap(gamma, BFM_GAMMA_SIZE);

        // calculate vertex
        Matrix<T, -1, -1> vertex_0 = (*faceModel).getShapeMeanBlock(3 * vertex_indices(0), 3).cast<T>()
            + (*faceModel).getExpMeanBlock(3 * vertex_indices(0), 3).cast<T>()
            + (*faceModel).getShapeBasisRowBlock(3 * vertex_indices(0), 3) * alphaMap
            + (*faceModel).getExpBasisRowBlock(3 * vertex_indices(0), 3) * gammaMap;

        Matrix<T, -1, -1> vertex_1 = (*faceModel).getShapeMeanBlock(3 * vertex_indices(1), 3).cast<T>()
            + (*faceModel).getExpMeanBlock(3 * vertex_indices(1), 3).cast<T>()
            + (*faceModel).getShapeBasisRowBlock(3 * vertex_indices(1), 3) * alphaMap
            + (*faceModel).getExpBasisRowBlock(3 * vertex_indices(1), 3) * gammaMap;

        Matrix<T, -1, -1> vertex_2 = (*faceModel).getShapeMeanBlock(3 * vertex_indices(2), 3).cast<T>()
            + (*faceModel).getExpMeanBlock(3 * vertex_indices(2), 3).cast<T>()
            + (*faceModel).getShapeBasisRowBlock(3 * vertex_indices(2), 3) * alphaMap
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

        T depth = alpha_bary * vertex_0_screen(2, 0) + beta_bary * vertex_1_screen(2, 0) + gamma_bary * vertex_2_screen(2, 0);

        // perspective-correct barycentric weights
        T d = alpha_bary * one_over_z0 + beta_bary * one_over_z1 + gamma_bary * one_over_z2;
        d = T(1) / d;

        alpha_bary *= d * one_over_z0;
        beta_bary *= d * one_over_z1;
        gamma_bary *= d * one_over_z2;

        residuals[0] = (depth - depth_source) * T(pointWeight);
        return true;
    }
private:
    const double pointWeight, depth_source, pixel_row, pixel_col, width, height;
    const Vector3i vertex_indices;
    const FaceModel* faceModel;
    const Matrix4d perspective_matrix;
    const Vector3d barycentric_coords;
};

class ColorConsistencyEnergy {
public:
    ColorConsistencyEnergy(const double& weight, const Vector3d& sourceRGB, const Vector3i& vertexIndices, const Vector3d& vertexWeights, const FaceModel* faceModel,
        const VectorXd& sh_basis):
        //initialization
        m_weight{ weight },
        m_sourceRGB{ sourceRGB },
        m_vertexIndices{ vertexIndices },
        m_vertexWeights{ vertexWeights },
        m_faceModel{ faceModel },
        m_sh_basis{ sh_basis }
    { }

    template <typename T>
    bool operator()(const T* const beta, const T* const sh_coefficients, T* residuals) const {
        Map<const Matrix<T, -1, 1> > betaMap(beta, BFM_BETA_SIZE);
        Map<const Matrix<T, -1, 1> > shCoefficientsMap(sh_coefficients, 27);

        Matrix<T, -1, -1> vertex_0_color = (*m_faceModel).getColorMeanBlock(3 * m_vertexIndices(0), 3).cast<T>()
            + (*m_faceModel).getColorBasisRowBlock(3 * m_vertexIndices(0), 3) * betaMap;
        Matrix<T, -1, -1> vertex_1_color = (*m_faceModel).getColorMeanBlock(3 * m_vertexIndices(1), 3).cast<T>()
            + (*m_faceModel).getColorBasisRowBlock(3 * m_vertexIndices(1), 3) * betaMap;
        Matrix<T, -1, -1> vertex_2_color = (*m_faceModel).getColorMeanBlock(3 * m_vertexIndices(2), 3).cast<T>()
            + (*m_faceModel).getColorBasisRowBlock(3 * m_vertexIndices(2), 3) * betaMap;

        //Calculate the RGB value of barycentric point by weights
        T T_targetRGB[3];
        T T_sourceRGB[3];
        T irradiances[3];

        for (int i = 0; i < 3; ++i) {
            irradiances[i] = T(0);
            for (int j = 0; j < 9; ++j) {
                irradiances[i] += shCoefficientsMap(i*9 + j, 0) * T(m_sh_basis(j));
            }
            if (irradiances[i] > T(1)) {
                irradiances[i] = T(1);
            }
        }
        
        for (int m = 0; m < 3; m++) {
            T_targetRGB[m] = (vertex_0_color(m) * T(m_vertexWeights(0)) + vertex_1_color(m) * T(m_vertexWeights(1)) + vertex_2_color(m) * T(m_vertexWeights(2))) * irradiances[m] * T(255);
            T_sourceRGB[m] = T(m_sourceRGB[m]);
        }
        //Apply the color consistency formula
        residuals[0] = (T_sourceRGB[0] - T_targetRGB[0]) * T(m_weight);
        residuals[1] = (T_sourceRGB[1] - T_targetRGB[1]) * T(m_weight);
        residuals[2] = (T_sourceRGB[2] - T_targetRGB[2]) * T(m_weight);
        return true;
    }
private:
    const Vector3d& m_sourceRGB;
    const Vector3i& m_vertexIndices;
    const Vector3d& m_vertexWeights;
    const VectorXd& m_sh_basis;
    const FaceModel* m_faceModel;
    const double m_weight;
};


class Optimizer {

public:
    // maybe add some options later when we are done with the basic version. Like GN/LM, Cuda/Cpu....
    Optimizer(double _landmakrWeight = 1, double _embeddingWeight=1, double _colorWeight = 1, double _shapeRegWeight = 0.33, double _expRegWeight = 0.33, double _coloRegWeight = 0.33, int _maxIteration = 7) {
        landmarkWeight = _landmakrWeight;
        embeddingWeight = _embeddingWeight;
        colorWeight = _colorWeight;
        shapeRegWeight = _shapeRegWeight;
        expRegWeight = _expRegWeight;
        colorRegWeight = _coloRegWeight;
        maxIteration = _maxIteration;
    }

    // optimize for the parameters
    void optimize(Face& face) {
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
        VectorXd sh_coefficients = face.getSHCoefficients();

        unsigned numLandmarks = face.getFaceModel().getNumLandmarks();

        MatrixXd source_depth = img.getDepthMap();
        vector<MatrixXd> source_color = img.getRGB();

        for (int i = 0; i < maxIteration; ++i) {
            ceres::Problem problem;
            // LANDMARK TERM
            for (unsigned i = 0; i < numLandmarks; i++)
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<FeatureSimilarityEnergy, 2, BFM_ALPHA_SIZE, BFM_GAMMA_SIZE, 6>
                    (new FeatureSimilarityEnergy(landmarkWeight, img.getLandmark(i), &faceModel, faceModel.getLandmarkVertexIdx(i), face.getIntrinsics(), img.getWidth(), img.getHeight())),
                    nullptr, alpha.data(), gamma.data(), poseIncrement.getData()
                );
            // REGULARIZATION TERM
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
            // COLOR AND EMBEDDING TERMS
            if (i > 0) {
                render.clear_buffers();
                Matrix4f full_projection_matrix = face.getFullProjectionMatrix().cast<float>().transpose();
                VectorXf vertices = face.calculateVerticesDefault().cast<float>();
                VectorXf colors = face.calculateColorsDefault().cast<float>();
                VectorXf sh_coefficients_ = face.getSHCoefficients().cast<float>();
                render.render(face, full_projection_matrix, vertices, colors, sh_coefficients_);

                cv::Mat color_buffer = render.get_color_buffer();
                cv::Mat indices_buffer = render.get_pixel_triangle_buffer();
                cv::Mat bary_coords = render.get_pixel_bary_coord_buffer();
                cv::Mat normal_buffer = render.get_pixel_normal_buffer();

                for (unsigned i = 0; i < img.getHeight(); ++i) {
                    for (unsigned j = 0; j < img.getWidth(); ++j) {
                        cv::Vec3b color = color_buffer.at<cv::Vec3b>(i, j);
                        if (!(color[0] == 0 && color[1] == 0 && color[2] == 0)) {
                            // Add Point2Point term
                            Vector3i indices = faceModel.getTriangulationByRow(indices_buffer.at<int>(i, j));
                            Vector3d bary_coords_;
                            bary_coords_ << bary_coords.at<cv::Vec3d>(i, j)[0], bary_coords.at<cv::Vec3d>(i, j)[1], bary_coords.at<cv::Vec3d>(i, j)[2];
                            problem.AddResidualBlock(
                                new ceres::AutoDiffCostFunction<GeometryPoint2PointConsistencyEnergy, 1, BFM_ALPHA_SIZE, BFM_GAMMA_SIZE, 6>
                                (new GeometryPoint2PointConsistencyEnergy(embeddingWeight, indices, &faceModel,
                                    1. - source_depth(i, j) / 255., face.getIntrinsics(), double(j) + 0.5, double(i) + 0.5,
                                    double(img.getWidth()), double(img.getHeight()), bary_coords_)),
                                nullptr, alpha.data(), gamma.data(), poseIncrement.getData()
                            );
                            
                            // Add color term
                            Vector3d source_color_;
                            source_color_ << double(source_color[0](i, j)), double(source_color[1](i, j)), double(source_color[2](i, j));

                            cv::Vec3d normal = normal_buffer.at<cv::Vec3d>(i, j);
                            Vector3d normal_;
                            normal_ << normal[0], normal[1], normal[2];

                            Vector<double, 9> sh_basis;
                            for (int i = 0; i < 9; ++i) {
                                sh_basis(i) = SH_basis_function(normal_, i);
                            }

                            problem.AddResidualBlock(
                                new ceres::AutoDiffCostFunction<ColorConsistencyEnergy, 3, BFM_BETA_SIZE, 27>
                                (new ColorConsistencyEnergy(colorWeight, source_color_, indices, bary_coords_, &faceModel, sh_basis)),
                                nullptr, beta.data(), sh_coefficients.data()
                            );
                        }
                    }
                }
            }

            ceres::Solver::Options options;
            //options.dense_linear_algebra_library_type = ceres::CUDA;
            options.num_threads = omp_get_max_threads();
            options.max_num_iterations = 50;
            options.linear_solver_type = ceres::ITERATIVE_SCHUR;
            // options.preconditioner_type = ceres::JACOBI;
            options.minimizer_progress_to_stdout = true;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.BriefReport() << std::endl;
            // UPDATE PARAMS
            face.setAlpha(alpha);
            face.setGamma(gamma);
            face.setBeta(beta);
            face.setExtrinsics(PoseIncrement<double>::convertToMatrix(poseIncrement));
            face.setSHCoefficients(sh_coefficients);
        }
    }
private:
    double landmarkWeight, shapeRegWeight, expRegWeight, colorRegWeight, embeddingWeight, colorWeight;
    int maxIteration;
};