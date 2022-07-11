#pragma once
#include "Eigen.h"

/**
 * A wrapper class for the 3x3 intrinsics with 4 degrees of freedom. Assuming no distortion (axis skew)
 * Projection is only an interface to the underlying array (in constructor, no copy
 * of the input array is made).
 * Important: Input array needs to have a size of at least 4.
 */
template <typename T>
class Projection {
public:
    explicit Projection(T* const array) : m_array{ array } { }

    void setZero() {
        for (int i = 0; i < 4; ++i)
            m_array[i] = T(0);
    }

    T* getData() const {
        return m_array;
    }

    /**
     * Applies the pose increment onto the input point and produces transformed output point.
     * Important: The memory for both input parameters (input and output) needs to be reserved (i.e. on the stack)
     * inputPoint need to have at least 3 positions and outputPoints 2
     * beforehand).
     */
    void apply(T* inputPoint, T* outputPoint) const {
        // m_array[1,2,3,4] = [fx, fy, mx, my]
        outputPoint[0] = (m_array[0] * inputPoint[0] + m_array[2] * inputPoint[2]) / inputPoint[2];
        outputPoint[1] = (m_array[1] * inputPoint[1] + m_array[3] * inputPoint[2]) / inputPoint[2];
    }

    /**
     * Converts the m_array into instrinsic 3x3 matrix.
     */
    static Matrix3d convertToMatrix(const Projection<double>& projection) {
        // m_array[1,2,3,4] = [fx, fy, mx, my]
        double* projArr = projection.getData();
        Matrix3d matIntrinsics = Matrix3d::Zero();
        matIntrinsics(0, 0) = projArr[0];
        matIntrinsics(1, 1) = projArr[1];
        matIntrinsics(0, 2) = projArr[2];
        matIntrinsics(1, 2) = projArr[3];
        matIntrinsics(2, 2) = 1.0;
        return matIntrinsics;
    }

    static void intrinsicsMatTo4DoG(const Matrix3d& instrinsics, double* res) {
        res[0] = instrinsics(0, 0);
        res[1] = instrinsics(1, 1);
        res[2] = instrinsics(0, 2);
        res[3] = instrinsics(1, 2);
    }

private:
    T* m_array;
};