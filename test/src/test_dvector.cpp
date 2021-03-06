#include <iostream>

#include "ParK.hpp"
#include "catch.hpp"
#include "mpi.h"
#include "vec.hpp"

// import all
using namespace ParK;

TEST_CASE("DVector", "tests") {

    // defin shorthand
    using X = Test::Vec<double>;

    // define a data vector for each this_rank
    int this_rank;
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &this_rank);

    // check we are running on four procs
    if (comm_size != 4)
        abort();

    // define the data. This is equivalent to
    // a = 0 1  1 2  2 3  3 4 - 5
    X data_a = { 1.0 * this_rank, this_rank + 1.0 };

    // b = 0 2  2 3  4 4  6 5 - 6
    X data_b = { 2.0 * this_rank, this_rank + 2.0 };

    // init vector with the tail set to zero
    DVector<X, 1> a(MPI_COMM_WORLD, data_a, this_rank + 2);
    DVector<X, 1> b(MPI_COMM_WORLD, data_b, this_rank + 3);

    SECTION("check data") {
        // exec shift upwards
        a.shift(DMatrixBandType::UPPER);
        REQUIRE(a.head(0) == this_rank);
        REQUIRE(a.head(1) == this_rank + 1);
        int next_rank = (this_rank + 1 + comm_size) % comm_size;
        REQUIRE(a.other(0) == next_rank);
        REQUIRE(a.other(1) == next_rank + 1);
        REQUIRE(a.tail(0) == this_rank + 2);
    }

    SECTION("check data") {
        // exec shift downwards
        a.shift(DMatrixBandType::LOWER);
        REQUIRE(a.head(0) == this_rank);
        REQUIRE(a.head(1) == this_rank + 1);
        int next_rank = (this_rank - 1 + comm_size) % comm_size;
        REQUIRE(a.other(0) == next_rank);
        REQUIRE(a.other(1) == next_rank + 1);
        REQUIRE(a.tail(0) == this_rank + 2);
    }

    SECTION("check norm and dot product") {
        REQUIRE(norm(a) == 8.306623862918075);
        REQUIRE(norm(b) == 12.083045973594572);
        REQUIRE(a * b == 0 * 0 + 1 * 2 + 1 * 2 + 2 * 3 + 2 * 4 + 3 * 4 + 3 * 6 + 4 * 5 + 5 * 6);
    }

    SECTION("copy constructor from DVector") {
        DVector<X, 1> c = b;
        REQUIRE(c.head(0) == b.head(0));
        REQUIRE(c.head(1) == b.head(1));
        REQUIRE(c.tail(0) == b.tail(0));
        REQUIRE(c.dinfo().rank() == b.dinfo().rank());
        REQUIRE(c.dinfo().comm() == b.dinfo().comm());
        REQUIRE(c.dinfo().size() == b.dinfo().size());
    }

    SECTION("copy constructor from DVectorExpr") {
        DVector<X, 1> d = 3 * b + a * 2 - a / 2;
        REQUIRE(d.head(0) == 3 * b.head(0) + a.head(0) * 2 - a.head(0) / 2);
        REQUIRE(d.head(1) == 3 * b.head(1) + a.head(1) * 2 - a.head(1) / 2);
        REQUIRE(d.tail(0) == 3 * b.tail(0) + a.tail(0) * 2 - a.tail(0) / 2);
        REQUIRE(d.dinfo().rank() == b.dinfo().rank());
        REQUIRE(d.dinfo().comm() == b.dinfo().comm());
        REQUIRE(d.dinfo().size() == b.dinfo().size());
    }

    SECTION("in place assignment from DVectorExpr") {
        DVector<X, 1> d_add = a;
        d_add += a;
        REQUIRE(d_add.head(0) == 2 * a.head(0));
        REQUIRE(d_add.head(1) == 2 * a.head(1));
        REQUIRE(d_add.tail(0) == 2 * a.tail(0));

        DVector<X, 1> d_sub = a;
        d_sub -= a;
        REQUIRE(d_sub.head(0) == 0);
        REQUIRE(d_sub.head(1) == 0);
        REQUIRE(d_sub.tail(0) == 0);

        DVector<X, 1> d_mul = a;
        d_mul *= 2;
        REQUIRE(d_mul.head(0) == 2 * a.head(0));
        REQUIRE(d_mul.head(1) == 2 * a.head(1));
        REQUIRE(d_mul.tail(0) == 2 * a.tail(0));

        DVector<X, 1> d_div = a;
        d_div /= 2;
        REQUIRE(d_div.head(0) == 0.5 * a.head(0));
        REQUIRE(d_div.head(1) == 0.5 * a.head(1));
        REQUIRE(d_div.tail(0) == 0.5 * a.tail(0));
    }

    SECTION("DVector makes a copy of the data") {
        X             data = { 1, 2 };
        int           tail = 0;
        DVector<X, 1> d(MPI_COMM_WORLD, data, tail);
        d.head(0) = 5;
        d.tail(0) = 6;

        REQUIRE(d.head(0) == 5);
        REQUIRE(d.tail(0) == 6);

        REQUIRE(tail == 0);
        REQUIRE(data[0] == 1);
    }
    SECTION("broadcast tail") {
        X             data = { 1, 2 };
        int           tail = this_rank;
        DVector<X, 1> d(MPI_COMM_WORLD, data, tail);
        d.tail(0) = this_rank;
        d.bc_tail();
        REQUIRE(d.tail(0) == comm_size - 1);
    }
}
