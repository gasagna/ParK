#include "ParK.hpp"
#include "catch.hpp"
#include "vec.hpp"
#include <cmath>
#include <iostream>
#include <random>

using namespace ParK;

TEST_CASE("gmres-serial", "tests") {

    // see Trefthen and Bau for this test case
    int m = 200;

    // define shorthand
    using Vec = Test::Vec<double>;
    using Mat = Test::SqMat<double>;

    // define random device & distribution
    std::random_device         rd{};
    std::mt19937               gen{ rd() };
    std::normal_distribution<> d{ 0, 1 };

    // define matrix
    Mat A = Mat(m, 0.0);
    for (auto i = 0; i != m; i++) {
        A(i, i) = 2;
        for (auto j = 0; j != m; j++) {
            A(i, j) = A(i, j) + 0.5 * d(gen) / std::sqrt(m);
        }
    }

    // and exact solution
    Vec x = Vec(m);
    for (auto i = 0; i != m; i++)
        x[i] = d(gen);

    // obtain right hand side
    Vec b   = A * x;
    Vec sol = b;

    // solve in place. In 20 iterations the
    // residual should drop by a factor of 10^8
    auto opts = GMRESOptions(1e-8, 20, true);
    solve_gmres(A, sol, opts);

    // subtract exact solution, this should
    Vec err = sol - x;
    REQUIRE(norm(err) / norm(b) < 1e-8);
}

TEST_CASE("gmres-dvec", "tests") {

    // define shorthand
    using Vec = Test::Vec<double>;
    using Mat = Test::SqMat<double>;

    // define matrix
    Mat A = {{1, 2, 3, 4}, 2};

    // and exact solution
    Vec x = {1, 2};

    

    // obtain right hand side
    Vec b   = A * x;
    Vec sol = b;

    // solve in place. In 20 iterations the
    // residual should drop by a factor of 10^8
    auto opts = GMRESOptions(1e-8, 20, true);
    solve_gmres(A, sol, opts);

    // subtract exact solution, this should
    Vec err = sol - x;
    REQUIRE(norm(err) / norm(b) < 1e-8);
}