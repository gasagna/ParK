#include "ParK.hpp"
#include "catch.hpp"
#include "vec.hpp"
#include <cmath>
#include <iostream>
#include <random>

using namespace ParK;

TEST_CASE("gmres-serial", "tests") {

    // see Trefthen and Bau for this test case
    int m = 50;

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

    SECTION("classical solution") {
        // solve in place. In 20 iterations the
        // residual should drop by a factor of 10^8
        auto opts = GMRESOptions(1e-8, 20, true);
        solve_gmres(A, sol, opts);

        // subtract exact solution, this should
        Vec err = sol - x;
        REQUIRE(norm(err) / norm(b) < 1e-8);
    }

    // for large delta the solution of 
    //  min_x ||A*x - b|| s.t. ||x|| < delta
    // is the exact solution, with small residual
    SECTION("hookstep calculations large") {
        double delta = 100;
        // solve in place. In 20 iterations the
        // residual should drop like before
        auto opts = GMRESOptions(1e-8, 20, true);
        solve_gmres(A, sol, delta, opts);

        // subtract exact solution, this should
        Vec err = sol - x;
        REQUIRE(norm(err) / norm(b) < 1e-8);
    }

    // for small delta we need to respect the constraint
    SECTION("hookstep calculations small") {
        double delta = 1;
        auto opts = GMRESOptions(1e-10, 50, true);
        solve_gmres(A, sol, delta, opts);
        REQUIRE(std::fabs(norm(sol) - 1.0) < 1e-7);
    }
}

TEST_CASE("gmres-dvec", "tests") {

    // define shorthand
    using Vec = Test::Vec<double>;
    using Mat = Test::SqMat<double>;

    // define local and global matrix
    Mat A = {{1, 2, 3, 4}, 2};
    DMatrix<Vec, 0, Mat> Ad_UP(MPI_COMM_WORLD, A, DMatrixBandType::UPPER);
    DMatrix<Vec, 0, Mat> Ad_LO(MPI_COMM_WORLD, A, DMatrixBandType::LOWER);

    // now define the exact solution, locally, and globally
    Vec x = {1, 2};
    DVector<Vec, 0> xd(MPI_COMM_WORLD, x);

    // obtain right hand side bd = Ad * xd
    DVector<Vec, 0> bd_UP = Ad_UP * xd;
    DVector<Vec, 0> bd_LO = Ad_LO * xd;

    // now this should be the matrix
    // 1   3  -1   0   0   0   0   0
    // 2   4   0  -1   0   0   0   0
    // 0   0   1   3  -1   0   0   0
    // 0   0   2   4   0  -1   0   0
    // 0   0   0   0   1   3  -1   0
    // 0   0   0   0   2   4   0  -1
   // -1   0   0   0   0   0   1   3
    // 0  -1   0   0   0   0   2   4
    //
    // (or the lower band version)
    // times the vector
    // [1 2 1 2 1 2 1 2]
    // 
    // which is the vector
    // [6 8 6 8 6 8 6 8]
    // 
    // in both cases
    //
    // the same everywhere

    REQUIRE( bd_UP.head(0) == 6 );
    REQUIRE( bd_LO.head(0) == 6 );
    REQUIRE( bd_UP.head(1) == 8 );
    REQUIRE( bd_LO.head(1) == 8 );

    // now make a copy of bd and try to get x by solving
    // Ad * x = b
    DVector<Vec, 0> sol_UP = bd_UP;
    DVector<Vec, 0> sol_LO = bd_LO;

    // solve in place. This should quickly converge to machine accuracy 
    auto opts = GMRESOptions(1e-8, 2, true); 
    solve_gmres(Ad_UP, sol_UP, opts);
    solve_gmres(Ad_LO, sol_LO, opts);

    // subtract exact solution, this should be small
    DVector<Vec, 0> err_UP = sol_UP - xd;
    DVector<Vec, 0> err_LO = sol_LO - xd;

    REQUIRE(norm(err_UP) / norm(bd_UP) < 1e-8);
    REQUIRE(norm(err_LO) / norm(bd_LO) < 1e-8);
}