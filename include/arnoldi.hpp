#pragma once
#include <eigen3/Eigen/Dense>

namespace ParK {

template <
    typename MAT,
    typename VEC>
class ArnoldiCache {
public:
    const MAT&       mat;
    std::vector<VEC> Qs;
    Eigen::MatrixXd  H;

    ArnoldiCache(MAT const& _mat, VEC const& v)
        : mat(_mat) {
        // make a first vector
        VEC q1 = v / norm(v);
        Qs.push_back(q1);
    }

    // update the arnoldi vectors
    void update() {
        // current number of Arnoldi vectors
        int n = Qs.size();

        // create a new vector from matvec
        VEC v = mat * Qs[n - 1];

        // resize H to accommodate more entries
        // see http://eigen.tuxfamily.org/dox/
        // classEigen_1_1PlainObjectBase.html#a712c25be1652e5a64a00f28c8ed11462
        H.conservativeResize(n + 1, n);

        // set to zero the elements we have created, because the
        // function above leaves them unitialized and the main 
        // loop below will not cover all of them, because 
        // it only fills the last column
        for (auto j = 0; j != n; j++)
            H(n, j) = 0;

        // create new vector by successive orthogonalisation
        for (auto j = 0; j != n; j++) {
            H(j, n - 1) = v * Qs[j];
            v -= H(j, n - 1) * Qs[j];
        }
        H(n, n - 1) = norm(v);
        v /= H(n, n - 1);
        Qs.push_back(v);
    }

    // Overwrite the vector 'v' with a linear combination of the
    // Arnoldi vectors with coeffients defined in 'coeffs'
    template <typename COEFFS>
    void lincomb(VEC& v, const COEFFS& coeffs) {
        // we must check that we have the right number
        // of coefficients for the linear combination
        if (Qs.size() != coeffs.size()+1)
            throw(std::invalid_argument("you pillock!"));

        // construct linear combination of the arnoldi vectors
        v = Qs[0] * coeffs[0];
        for (auto i = 1; i != coeffs.size(); i++)
            v += Qs[i] * coeffs[i];
    }
};

} // namespace PaKr