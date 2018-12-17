#pragma once
#include "mpi.h"

namespace ParK {

/////////////////////////////////////////////////////////////////////////////////////////
// Specify whether the distributed matrix has a block of negative identities
// on the upper diagonal (UPPER), like so
//  0 -1  0  0
//  0  1 -1  0
//  0  0  2 -1
// -1  0  0  3
// or on the lower one (LOWER), like so
//  0  0  0 -1
// -1  1  0  0
//  0 -1  2  0
//  0  0 -1  3
enum class DMatrixBandType : int {
    UPPER,
    LOWER
};

class DInfo {
private:
    // available fields
    MPI_Comm _comm;
    int      _size;
    int      _rank;

public:
    // delete default constructor to avoid rubbish
    DInfo() = delete;

    // Only make it possible to construct from a communicator
    DInfo(MPI_Comm comm)
        : _comm(comm) {
        MPI_Comm_size(_comm, &_size);
        MPI_Comm_rank(_comm, &_rank);
    }

    // access fields through public interface
    MPI_Comm comm() const { return _comm; }
    int      size() const { return _size; }
    int      rank() const { return _rank; }
};

} // namespace PaKr