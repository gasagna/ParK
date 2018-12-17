#include <iostream>

#include "catch.hpp"
#include "mpi.h"
#include "vec.hpp"

TEST_CASE("Vec", "tests") {

    // define shorthand
    using Vec = Test::Vec<double>;

    // define some data
    Vec a = { 1.0 };
    Vec b = { 2.0 };
    Vec c = { 3.0 };
    Vec d = { 5.0 };

    d = 2 * a + b * 4 + c / 3;

    REQUIRE(a[0] == 1);
    REQUIRE(b[0] == 2);
    REQUIRE(c[0] == 3);
    REQUIRE(d[0] == 2 * a[0] + b[0] * 4 + c[0] / 3);

    d /= 2;
    REQUIRE(d[0] == a[0] + b[0] * 2 + c[0] / 6);

    d += a + 2 * b;
    REQUIRE(d[0] == a[0] + b[0] * 2 + c[0] / 6 + a[0] + 2 * b[0]);
}

TEST_CASE("MatVec", "tests") {

    // define shorthand
    using Vec = Test::Vec<double>;
    using Mat = Test::SqMat<double>;

    // define some data
    Vec x = { 1.0, 2.0 };
    Mat A = {{ 2.0, 3.0, 4.0, 5.0}, 2};

    // copy constructor from matvec
    Vec y = A*x;

    REQUIRE( y(0) == 2*1 + 4*2 );
    REQUIRE( y(1) == 3*1 + 5*2 );

    // copy assignement from matvec
    y = A*x;

    REQUIRE( y(0) == 2*1 + 4*2 );
    REQUIRE( y(1) == 3*1 + 5*2 );
}