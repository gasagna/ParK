#include "ParK.hpp"
#include "catch.hpp"
#include "vec.hpp"
#include <cmath>
#include <iostream>
#include <random>

using namespace ParK;

TEST_CASE("utils", "tests") {
    // couple of example functions with known zero
    auto f1 = [](double x) { return x*x - 2*x - 1; };
    auto f2 = [](double x) { return std::exp(x) - std::cos(x); };
    
    REQUIRE( std::fabs(fzero(f1, 3.0) - 2.414213562373095) < 1e-6 );
    REQUIRE( std::fabs(fzero(f2, 3.0) - 0) < 1e-6 );
}
