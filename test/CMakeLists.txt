cmake_minimum_required(VERSION 3.0.0)
project(runtests)

# include directories
include_directories(include)     # for catch
include_directories(../include)  # for PaKr

# set compiler
set(CXX "mpic++")

# add compiler flags
set(CXX_FLAGS "--std=c++17")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_FLAGS}")

# link to mpi libs
set(CMAKE_CXX_STANDARD_LIBRARIES -lmpi)

# will compile and link all file matching this pattern
file(GLOB TESTFILES "src/test_*.cpp")

# create executable
add_executable(runtests src/runtests.cpp ${TESTFILES})
target_link_libraries(runtests ${LINK_LIB})
