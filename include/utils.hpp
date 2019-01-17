#pragma once

namespace ParK {

// approximate derivative of f using a centered difference
template <typename F>
double der(F&& f, double x, double epsilon) {
    return (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon);
}

// bare bones newton-raphson algorithm 
template <typename F>
double fzero(F&&          f,               // function we wish to find a zero of
             double       x,               // initial guess
             const double ftol    = 1e-14, // tolerances and iteration parameters
             const double xtol    = 1e-14, //
             const int    maxiter = 20,    //
             double       epsilon = 1e-6) {     // step for derivative approximation

    for (int iter = 0; iter != maxiter; iter++) {
        // calc function, derivative and correction
        double fval  = f(x);
        double fpval = der(std::forward<F>(f), x, epsilon);
        double dx    = -fval / fpval;

        // apply correction
        x = x + dx;

        // exit checks
        if (std::fabs(fval) < ftol || std::fabs(dx) < xtol) return x;
    }

    throw std::runtime_error("iterations did not converge");
}
}