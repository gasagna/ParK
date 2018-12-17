#include "ParK.hpp"
#include "catch.hpp"
#include "vec.hpp"
#include <cmath>
#include <iostream>

using namespace ParK;

TEST_CASE("arnoldi", "tests") {

    // define shorthand
    using Vec = Test::Vec<double>;
    using Mat = Test::SqMat<double>;

    // define some data
    Vec x = { 1, 2, 3, 4 };
    Mat A = { { 1, 3, 2, 4, 5, 7, 6, 8, 9, 10, 12, 13, 11, 14, 16, 15 }, 4 };

    // use this for checking
    Vec expected = Vec(4);

    // define arnoldi iteration object
    auto arnit = ArnoldiCache(A, x);

    // in the following expected values have been computed separately
    // the first vector is simply q1 = x/norm(x).
    expected = { 0.18257418583505536,
        0.3651483716701107,
        0.5477225575051661,
        0.7302967433402214 };
    expected -= arnit.Qs[0];
    REQUIRE(norm(expected) < 1e-14);

    // the second vector is
    // v = A*q0 - ((A*q0)*q0)*q0)
    // q1 = v/norm(v)
    arnit.update();
    expected = { 0.7404414539553872,
        0.48013854990607846,
        0.05577919372485217,
        -0.4670140337355247 };
    expected -= arnit.Qs[1];
    REQUIRE(norm(expected) < 1e-14);

    // the third vector is
    // v = A*q1 - ((A*q1)*q1)*q1) - ((A*q1)*q0)*q0)
    // q2 = v/norm(v)
    arnit.update();
    expected = { -0.0005172534601315739,
        0.45894353702538104,
        -0.805536055244937,
        0.37480958628604544 };
    expected -= arnit.Qs[2];
    REQUIRE(norm(expected) < 1e-14);

    // the fourth vector is
    // v = A*q2 - ((A*q2)*q2)*q2) - ((A*q2)*q1)*q1) - ((A*q2)*q0)*q0)
    // q3 = v/norm(v)
    arnit.update();
    expected = { -0.6468483998433922,
        0.6523070361289901,
        0.21908981091742039,
        -0.32875877629171274 };
    expected -= arnit.Qs[3];
    REQUIRE(norm(expected) < 1e-14);

    REQUIRE(fabs(arnit.H(0, 0) - 36.86666666666666) < 1e-14);
    REQUIRE(fabs(arnit.H(1, 0) - 11.128741568069987) < 1e-14);
    REQUIRE(fabs(arnit.H(2, 0) - 0.0) < 1e-14);
    REQUIRE(fabs(arnit.H(3, 0) - 0.0) < 1e-14);
    REQUIRE(fabs(arnit.H(0, 1) + 1.3734207368331421) < 1e-14);
    REQUIRE(fabs(arnit.H(1, 1) + 1.6776908538481745) < 1e-14);
    REQUIRE(fabs(arnit.H(2, 1) - 1.9804529365004642) < 1e-14);
    REQUIRE(fabs(arnit.H(3, 1) - 0.0) < 1e-14);
    REQUIRE(fabs(arnit.H(0, 2) + 1.3690518077906155) < 1e-14);
    REQUIRE(fabs(arnit.H(1, 2) - 0.0770259326323658) < 1e-14);
    REQUIRE(fabs(arnit.H(2, 2) - 0.4815501434404343) < 1e-14);
    REQUIRE(fabs(arnit.H(3, 2) - 0.9887157600533213) < 1e-14);

    // now check we compute the linear combination properly
    Vec v          = Vec(4);
    Vec mustbezero = Vec(4);
    Vec coeffs     = { 1, 3, 4 };

    arnit.lincomb(v, coeffs);
    mustbezero = v - (arnit.Qs[0] * 1 + arnit.Qs[1] * 3 + arnit.Qs[2] * 4);
    REQUIRE(norm(mustbezero) < 1e-15);

    // we must pass a number of coefficien equal to the number of arnoldi vectors
    Vec coeffs_1     = { 1, 3, };
    Vec coeffs_2     = { 1, 3, 4, 5 };
    REQUIRE_THROWS( arnit.lincomb(v, coeffs_1) );
    REQUIRE_THROWS( arnit.lincomb(v, coeffs_2) );
}