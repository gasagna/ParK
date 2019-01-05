#pragma once
#include "mpi.h"
#include <array>
#include <cstddef>

#include "dinfo.hpp"
#include "dvector.hpp"

namespace ParK {

/////////////////////////////////////////////////////////////////////////////////////////
// Forward declaration
template <typename X, std::size_t NBORDER, typename OPER>
struct DMatVec;

/////////////////////////////////////////////////////////////////////////////////////////
// Class for a distributed matrix with bidiagonal, block cyclic structure,
// with right and bottom bordering vectors. This assume the bottom right
// corner is alwasy zero.
template <typename X, std::size_t NBORDER, typename OPER>
class DMatrix {
private:
    // alias
    using dvector = DVector<X, NBORDER>;

    // members
    DMatrixBandType      _btype;
    const DInfo          _dinfo;
    const OPER&          _oper;
    std::vector<dvector> _rborders;
    std::vector<dvector> _dborders;

public:
    // the constructor accepts a communicator, an operator used
    // to compute the diagonal submatrix/subvector products and
    // a distributed array, which is used to constructor the
    // bordering vectors by creating copies of it.
    DMatrix(MPI_Comm            comm,
            const OPER&         oper,
            DMatrixBandType     btype,
            const DVector<X, NBORDER>& seed)
        : _btype(btype)
        , _dinfo(comm)
        , _oper(oper)
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
    DMatVec<X, NBORDER, OPER> operator*(dvector& x) const {
        return { *this, x };
    }

    ////////////////////////////////////////////////////////////////
    // field accessors
    const DMatrixBandType btype() const { return _btype; }

    const OPER& oper() const { return _oper; }

    const DInfo& dinfo() const { return _dinfo; }

    // add bounds checking here, to avois surprises
    const dvector& rborder(const std::size_t i) const { return _rborders.at(i); }
    const dvector& dborder(const std::size_t i) const { return _dborders.at(i); }
    dvector&       rborder(const std::size_t i) { return _rborders.at(i); }
    dvector&       dborder(const std::size_t i) { return _dborders.at(i); }
};

////////////////////////////////////////////////////////////////
// Lazy matmul object that gets created when we execute A*x
template <typename X, std::size_t NBORDER, typename OPER>
struct DMatVec {
    ////////////////////////////////////////////////////////////////
    // Members
    const DMatrix<X, NBORDER, OPER>& _dmat;
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

        // then add other diagonal which is the product of
        // an identity matrix with the distributed vector
        _y.head() = _y.head() - _x.other();

        // now broadcast the tail from the last rank before the calcs
        // with the right bordering vectors. Maybe replace with RMA
        // in the tail(size_t) call
        _x.bc_tail();

        for (auto i = 0; i != NBORDER; i++) {
            // add terms due to the right bordering vectors
            _y.head() += _dmat.rborder(i).head() * _x.tail(i);

            // calculate dot product with lower bordering vector. This gets sent
            // to every rank, including the last one (which is where we need it)
            _y.tail(i) = _dmat.dborder(i) * _x;
        }
    }
};

} // namespace ParK