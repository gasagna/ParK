#pragma once
#include "mpi.h"
#include <array>

#include "dinfo.hpp"
#include "dvector.hpp"

namespace ParK {

/////////////////////////////////////////////////////////////////////////////////////////
// Forward declaration
template <
    typename OP,
    typename X,
    std::size_t NBORDER>
struct DMatVec;

/////////////////////////////////////////////////////////////////////////////////////////
// Class for a distribute matrix with bidiagonal, block cyclic structure
template <
    typename OPER,
    typename X,
    std::size_t NBORDER>
class DMatrix {
private:
    DMatrixBandType _btype;
    const DInfo     _dinfo;
    const OPER&     _oper;
    // std::array<DVector<X, NBORDER>, NBORDER> _rborder;
    // std::array<DVector<X, NBORDER>, NBORDER> _dborder;

public:
    // the constructor accepts a communicator, an operator used
    // to compute the diagonal submatrix/subvector products and
    // an array of right and lower bordering vectors.
    // DMatrix(MPI_Comm comm,
    //         OP &op,
    //         // std::array<DVector<X, NBORDER>, NBORDER> rborder,
    //         // std::array<DVector<X, NBORDER>, NBORDER> dborder,
    //         const bool isupper = false)
    //     : _isupper(isupper), _dinfo(comm), _op(op), _rborder(rborder), _dborder(dborder) {}

    // provide an optional constructor when NBORDER = 0
    DMatrix(MPI_Comm comm, const OPER& oper, DMatrixBandType btype)
        : _btype(btype)
        , _dinfo(comm)
        , _oper(oper) {
        static_assert(NBORDER == 0);
    }

    // When we take A*x, we return a lazy object that can be
    // used in e.g. a copy assignment operation on a DVector.
    // Note 'x' is not marked const, as we shift the data
    // up or down during the matrix-vector product
    DMatVec<OPER, X, NBORDER> operator*(DVector<X, NBORDER>& x) {
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
    // members
    const DMatrix<OPER, X, NBORDER>& _dmat;
    DVector<X, NBORDER>&             _x;

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

        // now broadcast the tail from the last rank with id=comm_size-1
        // WHY???
        MPI_Bcast(_x.tail().data(), NBORDER, MPI_DOUBLE,
            _dmat.dinfo().size() - 1, _dmat.dinfo().comm());

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

} // namespace PaKr