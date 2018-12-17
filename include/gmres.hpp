#pragma once

#include "arnoldi.hpp"

namespace PaKr {

////////////////////////////////////////////////////////////////
// Options for the solver
struct GMRESOptions {
    const double res_norm_tol;
    const int    maxiter;
    const bool   verbose;
    GMRESOptions(double _res_norm_tol = 1e-4,
        int             _maxiter      = 10,
        bool            _verbose      = true)
        : res_norm_tol(_res_norm_tol)
        , maxiter(_maxiter)
        , verbose(_verbose) {}
};

////////////////////////////////////////////////////////////////
template <
    typename MAT,
    typename VEC>
void solve_gmres(MAT& mat, VEC& b, GMRESOptions opts = GMRESOptions()) {
    // the arnoldi iteration cache
    ArnoldiCache<MAT, VEC> _arnit(mat, b);
    
    // right hand side of the least squares problem
    Eigen::VectorXd _g(1);
    
    // solution vector
    Eigen::VectorXd _y;
    
    // temporary vector for residual calculations
    Eigen::VectorXd _res;

    // init g to the norm of b, and use this later for termination conditions
    _g[0] = norm(b);

    // make sure we do not do rubbish
    if (_g[0] == 0)
        throw(std::invalid_argument("norm of right hand side is zero"));

    // start counting iterations from 1
    int iter = 1;

    // until conditions are verified below
    while (true) {
        // update arnoldi iteration object with a new vector
        _arnit.update();

        // increase size of g by adding a zero
        _g.conservativeResize(iter + 1);
        _g[iter] = 0;

        // solve least squares problem using QR decomposition.
        // This automatically reshapes y to the correct size
        _y = _arnit.H.colPivHouseholderQr().solve(_g);

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