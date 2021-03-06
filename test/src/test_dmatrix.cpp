#include <iostream>

#include "ParK.hpp"
#include "catch.hpp"
#include "mpi.h"
#include "vec.hpp"

// import all
using namespace ParK;

TEST_CASE("DMatrix", "tests") {

    // defin shorthand
    using Vec = Test::Vec<double>;

    // define a data vector for each this_rank
    int this_rank;
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &this_rank);

    // check we are running on four procs
    if (comm_size != 4)
        abort();

    // define the data
    auto data_in = Vec(1);
    if (this_rank == 0) data_in[0] = 1;
    if (this_rank == 1) data_in[0] = 2;
    if (this_rank == 2) data_in[0] = 3;
    if (this_rank == 3) data_in[0] = 4;

    SECTION("no borders") {
        SECTION("lower band") {
            // make a copy, where we will store the output
            Vec data_out = data_in;

            // input vector with the no tail
            DVector<Vec, 0> x(MPI_COMM_WORLD, data_in);
            DVector<Vec, 0> y = x;

            // define a distributed matrix
            DMatrix<Vec, 0, int> A(MPI_COMM_WORLD, this_rank, DMatrixBandType::LOWER);

            // exec product
            //
            //    / 0  0  0 -1\  / 1 \      / -4 \
            //   | -1  1  0  0 ||  2  | -> |   1  |
            //   |  0 -1  2  0 ||  3  | -> |   4  |
            //    \ 0  0 -1  3/  \ 4 /      \  9 /
            y = A * x;

            // compare with expected value
            if (this_rank == 0)
                REQUIRE(y.head()[0] == -4);
            if (this_rank == 1)
                REQUIRE(y.head()[0] == 1);
            if (this_rank == 2)
                REQUIRE(y.head()[0] == 4);
            if (this_rank == 3)
                REQUIRE(y.head()[0] == 9);
        }

        SECTION("upper band") {
            // make a copy, where we will store the output
            Vec data_out = data_in;

            // input vector with the no tail
            DVector<Vec, 0> x(MPI_COMM_WORLD, data_in);
            DVector<Vec, 0> y = x;

            // define a distributed matrix
            DMatrix<Vec, 0, int> A(MPI_COMM_WORLD, this_rank, DMatrixBandType::UPPER);

            // exec product
            //
            //    /  0 -1  0  0\  / 1 \      / -2 \
            //   |   0  1 -1  0 ||  2  | -> |  -1  |
            //   |   0  0  2 -1 ||  3  | -> |   2  |
            //    \ -1  0  0  3/  \ 4 /      \ 11 /
            y = A * x;

            // compare with expected value
            if (this_rank == 0)
                REQUIRE(y.head()[0] == -2);
            if (this_rank == 1)
                REQUIRE(y.head()[0] == -1);
            if (this_rank == 2)
                REQUIRE(y.head()[0] == 2);
            if (this_rank == 3)
                REQUIRE(y.head()[0] == 11);
        }
    }

    SECTION("with borders") {
        SECTION("one border") {
            // make a copy, where we will store the output
            Vec data_out = data_in;

            // input vector with tail
            DVector<Vec, 1> x(MPI_COMM_WORLD, data_in, 5);

            // allocate solution making a copy
            DVector<Vec, 1> y = x;

            // define a distributed matrix
            DMatrix<Vec, 1, int> A(MPI_COMM_WORLD,
                                   this_rank,
                                   DMatrixBandType::LOWER,
                                   y);

            // fill borders. Note we need to fill the tail too!
            A.dborder(0).head()  = this_rank;
            A.rborder(0).head()  = this_rank + 1;
            A.dborder(0).tail(0) = 0;
            A.rborder(0).tail(0) = 0;

            // exec product
            //
            //    / 0  0  0 -1  1\  / 1 \      /  1 \
            //   | -1  1  0  0  2 ||  2  | -> |  11  |
            //   |  0 -1  2  0  3 ||  3  | -> |  19  |
            //   |  0  0 -1  3  4 ||  4  |    |  29  |
            //   \  0  1  2  3  0/ \  5  /    \  20  /
            y = A * x;

            // compare with expected value
            if (this_rank == 0)
                REQUIRE(y.head(0) == 1);
            if (this_rank == 1)
                REQUIRE(y.head(0) == 11);
            if (this_rank == 2)
                REQUIRE(y.head(0) == 19);
            if (this_rank == 3)
                REQUIRE(y.head(0) == 29);

            // note the tail gets updated everywhere
            REQUIRE(y.tail(0) == 20);
        }
    }
}