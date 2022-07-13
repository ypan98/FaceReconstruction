#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "Face.h"
#include "PoseIncrement.h"
#include <stdio.h>

using namespace std;

/*
* Feature Similarity (Landmarks) energy optimization.
* Truly denser problem: every residual depends on every parameter alpha, beta, extrinsic and intrinsic. => No big difference defining a residual block per point or
* the whole as a big block. By means of code consistency we still do it by point.
*/
class FeatureSimilarityEnergy {
public:
    FeatureSimilarityEnergy(const double& _landmarkWeight, const Vector2d& _landmark, const FaceModel* _faceModel, const unsigned& _vertexIdx) :
        landmarkWeight(_landmarkWeight),
        landmark(_landmark),
        faceModel(_faceModel),
        vertexIdx(_vertexIdx)
    { }

    template <typename T>
    bool operator()(const T const* alpha, const T const* gamma, const T* const poseFormatExtrinsics, const T const* intrinsics, T* residuals) const {
        // ************* USING EIGEN (doesnt even compile) *************??
        /*Map<const Matrix<T, Dynamic, 1>> constAlphaMap(alpha, BFM_ALPHA_SIZE);
        Map<const Matrix<T, Dynamic, 1>> constGammaMap(gamma, BFM_GAMMA_SIZE);
        Vector3d vertex = ((*faceModel).getShapeMean().block(3 * vertexIdx, 0, 3, 1)
            + (*faceModel).getExpMean().block(3 * vertexIdx, 0, 3, 1)) 
            + (*faceModel).getShapeBasisStdDivided().middleRows(3 * vertexIdx, 3) * constAlphaMap
            + (*faceModel).getExpBasisStdDivided().middleRows(3 * vertexIdx, 3) * constGammaMap;*/

        // ************* MAKING CALCULATIONS WITH LOOP *************
        T vertex[3];
        // mean
        for (unsigned i = 0; i < 3; i++) vertex[i] = T((*faceModel).getShapeMeanElem(3 * vertexIdx + i)) + T((*faceModel).getExpMeanElem(3 * vertexIdx + i));
        // shape basis
        for (unsigned i = 0; i < BFM_ALPHA_SIZE; i++) 
            for (unsigned j = 0; j < 3; j++) vertex[j] += T((*faceModel).getShapeBasisStdMultipliedElem(3*vertexIdx + j, i)) * alpha[i];
        // expression basis
        for (unsigned i = 0; i < BFM_GAMMA_SIZE; i++)
            for (unsigned j = 0; j < 3; j++) vertex[j] += T((*faceModel).getExpBasisStdMultipliedElem(3*vertexIdx + j, i)) * gamma[i];
        // apply pose (extrinsics)
        T vertex_transformed[3];
        PoseIncrement<T> pose_inc(const_cast<T* const>(poseFormatExtrinsics));
        pose_inc.apply(vertex, vertex_transformed);
        // apply intrinsics and divide by z
        T vertex_pixel_coord[2];
        vertex_pixel_coord[0] = (intrinsics[0] * vertex_transformed[0] + intrinsics[2] * vertex_transformed[2])/ vertex_transformed[2];
        vertex_pixel_coord[1] = (intrinsics[1] * vertex_transformed[1] + intrinsics[3] * vertex_transformed[2])/ vertex_transformed[2];
        residuals[0] = (T(landmark(0)) - vertex_pixel_coord[0])*T(landmarkWeight);
        residuals[1] = (T(landmark(1)) - vertex_pixel_coord[1])*T(landmarkWeight);
        return true;
    }

private:
    const double landmarkWeight;
    const FaceModel* faceModel;
    const Vector2d landmark;
    const unsigned vertexIdx;
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

class GeometryConsistencyEnergy {
public:
    GeometryConsistencyEnergy(const double& _geometryWeight, const VectorXf& alpha, const VectorXf& gamma, const Vector3i& vertex_indices, const FaceModel* _faceModel,
        const double& depth_source, const Matrix4d& projection_matrix) :
        _geometryWeight(_geometryWeight),
        alpha(alpha),
        gamma(gamma),
        vertex_indices(vertex_indices),
        _faceModel(_faceModel),
        depth_source(depth_source),
        projection_matrix(projection_matrix)
    { }

    template <typename T>
    bool operator() (const T const* alpha, const T const* gamma) const{
        Vector4d 
        // mean
        for (unsigned i = 0; i < 3; i++) vertex[i] = T((*faceModel).getShapeMeanElem(3 * vertexIdx + i)) + T((*faceModel).getExpMeanElem(3 * vertexIdx + i));
        // shape basis
        for (unsigned i = 0; i < BFM_ALPHA_SIZE; i++)
            for (unsigned j = 0; j < 3; j++) vertex[j] += T((*faceModel).getShapeBasisStdMultipliedElem(3 * vertexIdx + j, i)) * alpha[i];
        // expression basis
        for (unsigned i = 0; i < BFM_GAMMA_SIZE; i++)
            for (unsigned j = 0; j < 3; j++) vertex[j] += T((*faceModel).getExpBasisStdMultipliedElem(3 * vertexIdx + j, i)) * gamma[i];


    }
private:
    const double _geometryWeight, depth_source;
    const VectorXf alpha, gamma;
    const Vector3i vertex_indices;
    const FaceModel* _faceModel;
    const Matrix4d projection_matrix;
};
class Optimizer {

public:
    // maybe add some options later when we are done with the basic version. Like GN/LM, Cuda/Cpu....
    Optimizer(double _landmakrWeight = 1, double _shapeRegWeight = 1, double _expRegWeight = 1, double _coloRegWeight = 1) {
        landmarkWeight = _landmakrWeight;
        shapeRegWeight = _shapeRegWeight;
        expRegWeight = _expRegWeight;
        colorRegWeight = _coloRegWeight;
    }

    // optimize for the parameters
    void optimize(Face& face) {
        // PREPARATION
        ceres::Problem problem;
        double pose[6] = {0};
        extrinsicsMatTo6DoG(face.getExtrinsics(), pose);
        PoseIncrement poseIncrement = PoseIncrement<double>(pose);
        Image img = face.getImage();
        FaceModel faceModel = face.getFaceModel();
        VectorXd alpha = face.getAlpha();
        VectorXd beta = face.getBeta();
        VectorXd gamma = face.getGamma();
        double intrinsics[4];   // for the intrinsics, we only need fx,fy,mx,my. Assuming distortion=0
        intrinsics[0] = face.getIntrinsics()(0,0);
        intrinsics[1] = face.getIntrinsics()(1,1);
        intrinsics[2] = face.getIntrinsics()(0,2);
        intrinsics[3] = face.getIntrinsics()(1,2);
        unsigned numLandmarks = face.getFaceModel().getNumLandmarks();
        // LANDMARK TERM
        for (unsigned i = 0; i < numLandmarks; i++) 
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<FeatureSimilarityEnergy, 2, BFM_ALPHA_SIZE, BFM_GAMMA_SIZE, 6, 4>
                (new FeatureSimilarityEnergy(landmarkWeight, img.getLandmark(i), &faceModel, faceModel.getLandmarkVertexIdx(i))),
                nullptr, alpha.data(), gamma.data(), poseIncrement.getData(), intrinsics
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
        // GEOMETRY TERM
        // ...
        // COLOR TERM
        // ...
        // SOLVE OPTIMIZATION
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.num_threads = 16;
        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << std::endl;
        // UPDATE PARAMS
        face.setAlpha(alpha);
        face.setGamma(gamma);
        face.setExtrinsics(PoseIncrement<double>::convertToMatrix(poseIncrement));
        face.setIntrinsics(constructInstrinsicsMat(intrinsics));
    }

private:
    double landmarkWeight, shapeRegWeight, expRegWeight, colorRegWeight;

    void extrinsicsMatTo6DoG(const Matrix4d& extrinsics, double* res) {
        Matrix3d mat = extrinsics.block(0, 0, 3, 3);
        Vector3d eulerAngles = mat.eulerAngles(0, 1, 2);
        res[0] = eulerAngles(0);
        res[1] = eulerAngles(1);
        res[2] = eulerAngles(2);
        res[3] = extrinsics(0, 3);
        res[4] = extrinsics(1, 3);
        res[5] = extrinsics(2, 3);
    }

    Matrix3d constructInstrinsicsMat(double* intrinsics) {
        Matrix3d matIntrinsics = Matrix3d::Zero();
        matIntrinsics(0, 0) = intrinsics[0];
        matIntrinsics(1, 1) = intrinsics[1];
        matIntrinsics(0, 2) = intrinsics[2];
        matIntrinsics(1, 2) = intrinsics[3];
        matIntrinsics(2, 2) = 1.0;
        return matIntrinsics;
    }
};