#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "Face.h"
#include "PoseIncrement.h"
#include "Projection.h"
#include <stdio.h>
#include <omp.h>

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
    bool operator()(const T const* alpha, const T const* gamma, const T* const extrinsicsArr, const T const* intrinsicsArr, T* residuals) const {
        // ************* USING EIGEN (very slow...) ***************
        Map<const Matrix<T, -1, 1> > alphaMap(alpha, BFM_ALPHA_SIZE);
        Map<const Matrix<T, -1, 1> > gammaMap(gamma, BFM_GAMMA_SIZE);
        // calculate vertex
        Matrix<T, -1, -1> vertex = (*faceModel).getShapeMeanBlock(3 * vertexIdx, 3).cast<T>()
            + (*faceModel).getExpMeanBlock(3 * vertexIdx, 3).cast<T>()
            + (*faceModel).getShapeBasisRowBlock(3 * vertexIdx, 3) * alphaMap
            + (*faceModel).getExpBasisRowBlock(3 * vertexIdx, 3) * gammaMap;

        // ************* MAKING CALCULATIONS WITH LOOP *************
        //T vertex[3];
        //// mean
        //for (unsigned i = 0; i < 3; i++) vertex[i] = T((*faceModel).getShapeMeanElem(3 * vertexIdx + i)) + T((*faceModel).getExpMeanElem(3 * vertexIdx + i));
        //// shape basis
        //for (unsigned i = 0; i < BFM_ALPHA_SIZE; i++) 
        //    for (unsigned j = 0; j < 3; j++) vertex[j] += T((*faceModel).getShapeBasisElem(3*vertexIdx + j, i)) * alpha[i];
        //// expression basis
        //for (unsigned i = 0; i < BFM_GAMMA_SIZE; i++)
        //    for (unsigned j = 0; j < 3; j++) vertex[j] += T((*faceModel).getExpBasisElem(3*vertexIdx + j, i)) * gamma[i];

        // apply pose (extrinsics)
        T vertex_transformed[3];
        PoseIncrement<T> pose_inc(const_cast<T* const>(extrinsicsArr));
        pose_inc.apply(vertex.data(), vertex_transformed);
        // apply projection (instrinsics)
        T vertex_pixel_coord[2];
        Projection<T> proj(const_cast<T* const>(intrinsicsArr));
        proj.apply(vertex_transformed, vertex_pixel_coord);
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
        // wrapped extrinsics, intrinsics. With 4 and 6 degrees of freedom respectively
        double poseArr[6]; // reserve space
        PoseIncrement<double>::extrinsicsMatTo6DoG(face.getExtrinsics(), poseArr);
        PoseIncrement poseIncrement = PoseIncrement<double>(poseArr);
        double projArr[4];   // reserve space
        Projection<double>::intrinsicsMatTo4DoG(face.getIntrinsics(), projArr);
        Projection projection = Projection<double>(projArr);
        Image img = face.getImage();
        FaceModel faceModel = face.getFaceModel();
        VectorXd alpha = face.getAlpha();
        VectorXd beta = face.getBeta();
        VectorXd gamma = face.getGamma();
        unsigned numLandmarks = face.getFaceModel().getNumLandmarks();
        // LANDMARK TERM
        for (unsigned i = 0; i < numLandmarks; i++) 
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<FeatureSimilarityEnergy, 2, BFM_ALPHA_SIZE, BFM_GAMMA_SIZE, 6, 4>
                (new FeatureSimilarityEnergy(landmarkWeight, img.getLandmark(i), &faceModel, faceModel.getLandmarkVertexIdx(i))),
                nullptr, alpha.data(), gamma.data(), poseIncrement.getData(), projection.getData()
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
        options.dense_linear_algebra_library_type = ceres::CUDA;
        options.num_threads = omp_get_max_threads();
        options.max_num_iterations = 500;
        options.linear_solver_type = ceres::DENSE_QR;
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
        face.setIntrinsics(Projection<double>::convertToMatrix(projection));
    }

private:
    double landmarkWeight, shapeRegWeight, expRegWeight, colorRegWeight;
};