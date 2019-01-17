#pragma once

#include "arnoldi.hpp"

namespace ParK {

////////////////////////////////////////////////////////////////
// Options for the solver
struct GMRESOptions {
    const double res_norm_tol;
    const int    maxiter;
    const bool   verbose;
    GMRESOptions(double _res_norm_tol = 1e-4,
                 int    _maxiter      = 10,
                 bool   _verbose      = true)
        : res_norm_tol(_res_norm_tol)
        , maxiter(_maxiter)
        , verbose(_verbose) {}
};

////////////////////////////////////////////////////////////////
template <typename MAT, typename VEC>
void solve_gmres(MAT& mat, VEC& b, GMRESOptions opts = GMRESOptions()) {
    _solve_gmres_impl(mat, b, 0.0, false, opts);
}

template <typename MAT, typename VEC>
void solve_gmres(MAT& mat, VEC& b, double delta, GMRESOptions opts = GMRESOptions()) {
    _solve_gmres_impl(mat, b, delta, true, opts);
}

template <typename MAT, typename VEC>
void _solve_gmres_impl(MAT& mat, VEC& b, double delta, bool solve_hookstep, GMRESOptions opts) {
    // the arnoldi iteration cache
    ArnoldiCache<MAT, VEC> _arnit(mat, b);

    // right hand side of the least squares problem
    Eigen::VectorXd _g(1);

    // solution vector
    Eigen::VectorXd _y;

    // temporary vector for residual calculations
    Eigen::VectorXd _res;

    // temporary vectors for hookstep calculations
    Eigen::VectorXd _p, _q, _d;

    // structure to perform the svd
    Eigen::BDCSVD<Eigen::MatrixXd> _svd;

    // init g to the norm of b, and use this later for termination conditions
    _g[0] = norm(b);

    // make sure we do not do rubbish
    if (_g[0] == 0)
        throw(std::invalid_argument("norm of right hand side is zero"));

    // start counting iterations from 1
    int iter = 1;

    // Define function to calculate the hookstep using a closure.
    // This calculates the norm of the vector _q by a loop, to 
    // avoid temporaries. It also writes over _q, so at the end 
    // of the optimisation we have it already.
    auto fun = [&](double mu) {
        double S = 0;
        for (auto i = 0; i != _q.size(); i++) {
            _q[i] = _p[i] * _d[i] / (mu + _d[i] * _d[i]);
            S += _q[i] * _q[i];
        }
        return std::sqrt(S) - delta;
    };

    // until conditions are verified below
    while (true) {
        // update arnoldi iteration object with a new vector
        _arnit.update();

        // increase size of g by adding a zero
        _g.conservativeResize(iter + 1);
        _g[iter] = 0;

        // increase size of _q to appropriate size
        _q.conservativeResize(iter);

        // compute svd. This can be used to solve both the hookstep 
        // problem or the least squares problem
        _svd.compute(_arnit.H, Eigen::ComputeThinV | Eigen::ComputeThinU);

        if (solve_hookstep) {
            // solve optimally constrained  problem
            // see Viswanath https://arxiv.org/pdf/0809.1498.pdf
            _p = _svd.matrixU().transpose() * _g;
            _d = _svd.singularValues();

            // Find mu > 0 such that ||q|| = delta. This is a hack 
            // to save few lines of bad looking code. If fun(0) < 0,
            // typically when delta is large, then q_i = p_i / d_i
            // otherwise q_i = p_i * d_i / (mu + d_i * d_i) with mu
            // such that ||q|| == delta
            if (fun(0) > 0) fzero(fun, 0);

            // construct vector y = V*q that generates the hookstep
            _y = _svd.matrixV() * _q;
        } else {
            // solve least squares problem using QR decomposition.
            // This automatically reshapes y to the correct size
            _y = _svd.solve(_g);
        }
        // std::cout << "y: " << _y.cols() << " " << _y.cols() << "\n";

        // get current residual vector
        _res = _arnit.H * _y - _g;

        // check tolerances
        if (_res.norm() < opts.res_norm_tol * _g[0])
            break;
        if (iter >= opts.maxiter)
            break;

        iter++;
    }

    // finally construct linear combination of Arnoldi vectors
    _arnit.lincomb(b, _y);
}

} // namespace PaKr