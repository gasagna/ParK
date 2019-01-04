#pragma once
#include "mpi.h"
#include <array>
#include <cstddef>

#include "dinfo.hpp"
#include "dvector.hpp"

namespace ParK {

/////////////////////////////////////////////////////////////////////////////////////////
// Forward declaration
template <typename X, std::size_t NBORDER, typename OP>
struct DMatVec;

/////////////////////////////////////////////////////////////////////////////////////////
// Class for a distributed matrix with bidiagonal, block cyclic structure,
// with right and bottom bordering vectors.
template <typename X, std::size_t NBORDER, typename OPER>
class DMatrix {
private:
    DMatrixBandType                  _btype;
    const DInfo                      _dinfo;
    const OPER&                      _oper;
    std::vector<DVector<X, NBORDER>> _rborders;
    std::vector<DVector<X, NBORDER>> _dborders;

public:
    // the constructor accepts a communicator, an operator used
    // to compute the diagonal submatrix/subvector products and
    // a distributed array, which is used to constructor the 
    // bordering vectors by creating copies of it.
    DMatrix(MPI_Comm            comm,
            const OPER&         oper,
            DMatrixBandType     btype,
            DVector<X, NBORDER> seed)
        : _isupper(isupper)
        , _dinfo(comm)
        , _op(op)
        , _rborders(NBORDER, seed)
        , _dborders(NBORDER, seed) {}

    // provide an optional constructor when NBORDER = 0
    DMatrix(MPI_Comm        comm,
            const OPER&     oper,
            DMatrixBandType btype)
        : _btype(btype)
        , _dinfo(comm)
        , _oper(oper) {
        static_assert(NBORDER == 0);
    }

    // When we take A*x, we return a lazy object that can be
    // used in e.g. a copy assignment operation on a DVector.
    // Note 'x' is not marked const, as we shift the data
    // up or down during the matrix-vector product
    DMatVec<OPER, X, NBORDER> operator*(DVector<X, NBORDER>& x) const {
        return { *this, x };
    }

    ////////////////////////////////////////////////////////////////
    // field accessors
    const DMatrixBandType btype() const {
        return _btype;
    }

    const OPER& oper() const {
        return _oper;
    }

    const DInfo& dinfo() const {
        return _dinfo;
    }
};

////////////////////////////////////////////////////////////////
// Lazy matmul object that gets created when we execute A*x
template <
    typename OPER,
    typename X,
    std::size_t NBORDER>
struct DMatVec {
    ////////////////////////////////////////////////////////////////
    // Members
    const DMatrix<OPER, X, NBORDER>& _dmat;
    const DVector<X, NBORDER>&       _x;

    ////////////////////////////////////////////////////////////////
    // actually executes y = A*x
    void execute(DVector<X, NBORDER>& _y) {
        // start non blocking transfer in a direction that
        // depends on whether the matrix has a block of identities
        // on the upper or lower block diagonal. If it's on the upper
        // block diagonal, we need to move data "upwards" in the
        // distributed vector _x
        _x.shift_init(_dmat.btype());

        // perform local products while transferring
        _y.head() = _dmat.oper() * _x.head();

        // wait for the transfer to finish
        _x.shift_wait();

        // then add other diagonal which the product of
        // an identity matrix with the dostrivuted vector
        _y.head() = _y.head() - _x.other();

        // now broadcast the tail from the last rank with id=comm_size-1 to all procs
        // because we need it to include the right bordering vectors in the
        MPI_Bcast(_x.tail().data(), NBORDER, MPI_DOUBLE, _dmat.dinfo().size() - 1, _dmat.dinfo().comm());

        // for (auto i = 0; i != NBORDER; i++)
        // {
        //     // and add terms due to the right bordering vectors
        //     _y.local() += _dmat._rborder[i].local() * _x.tail()[i];

        //     // calculate dot product with lower bordering vector. This gets sent
        //     // to every rank, including the last one (which is where we need it)
        //     _y.tail[i] = _dmat._dborder[i] * _x;
        // }
    }
};

} // namespace ParK