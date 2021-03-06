#pragma once
#include "mpi.h"
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <tuple>

#include "dinfo.hpp"

namespace ParK {

/////////////////////////////////////////////////////////////////////////////////////////
// Forward declaration
template <typename X, std::size_t NBORDER, typename OP>
struct DMatVec;

/////////////////////////////////////////////////////////////////////////////////////////
// Base class for expression templates
template <typename E>
struct DVectorExpr {

    auto& tail() const { return static_cast<const E&>(*this).tail(); }
    auto& head() const { return static_cast<const E&>(*this).head(); }
    auto& other() const { return static_cast<const E&>(*this).other(); }
    auto& dinfo() const { return static_cast<const E&>(*this).dinfo(); }

    auto tail(const std::size_t i) const { return static_cast<const E&>(*this).tail(i); }
    auto head(const std::size_t i) const { return static_cast<const E&>(*this).head(i); }
    auto other(const std::size_t i) const { return static_cast<const E&>(*this).other(i); }
};

/////////////////////////////////////////////////////////////////////////////////////////
// Distributed vector class for linear problems with bidiagonal, block cyclic structure
template <typename X, std::size_t NBORDER>
class DVector : public DVectorExpr<DVector<X, NBORDER>> {
private:
    MPI_Request                 _requests[2];
    MPI_Status                  _statuses[2];
    const DInfo                 _dinfo;
    X                           _head;
    X                           _other;
    std::array<double, NBORDER> _tail = { 0.0 };

public:
    /***********************************************************************************/
    // Copy constructor and assignement
    /***********************************************************************************/

    // Copy constructor from communicator, data and tail elements (makes a copy of data)
    template <typename... TAIL>
    DVector(MPI_Comm comm, X& x, TAIL... tail)
        : _dinfo(comm)
        , _head(x)
        , _other(similar(_head))
        , _tail({ static_cast<double>(tail)... }) {
        static_assert(sizeof...(TAIL) == NBORDER,
                      "variadic input must be consistent with the template parameter.");
    }

    // Copy constructor from a DMatVec
    // This function is provided for convenience. We first initialize
    // the DInfo struct from the communicator object of the operator
    // to make sure we do not have rubbish in the communication code.
    // We then initialise the 'head' and 'other' members. The 'tail'
    // is automatically initialized by the compiler and is filled
    // by the call to 'execute'
    template <typename OPER>
    DVector(DMatVec<X, NBORDER, OPER>&& m)
        : _dinfo(m._x.dinfo().comm())
        , _head(similar(m._x.head()))
        , _other(similar(m._x.head())) {
        m.execute(*this);
    }

    // Copy constructor from a DVectorExpr
    template <typename E>
    DVector(const DVectorExpr<E>& a)
        : _dinfo(a.dinfo().comm())
        , _head(similar(a.head()))
        , _other(similar(a.head())) {
        for (auto i = 0; i != _head.size(); i++)
            _head[i] = a.head(i);

        for (auto i = 0; i != _tail.size(); i++)
            _tail[i] = a.tail(i);

        // we do not initialised 'other', and require an explicit 'shift' to move data.
    }

    // Copy assignment from a lazy matmul object
    template <typename OPER>
    DVector<X, NBORDER>& operator=(DMatVec<X, NBORDER, OPER>&& m) {
        m.execute(*this);
        return *this;
    }

// Define in place assignment operations such as +=, -= and = with
// DVectorExpr objects and *=, /= and = with scalars. This is
// because we treat DVector as an element of a vector space, and
// a vector plus a scalar means nothing
#define _DEFINE_INPLACE_OP_DVECEXPR(_Op)                         \
    template <typename E>                                        \
    DVector<X, NBORDER>& operator _Op(const DVectorExpr<E>& a) { \
        for (auto i = 0; i != _head.size(); i++)                 \
            _head[i] _Op a.head(i);                              \
                                                                 \
        for (auto i = 0; i != _tail.size(); i++)                 \
            _tail[i] _Op a.tail(i);                              \
                                                                 \
        return *this;                                            \
    }
    _DEFINE_INPLACE_OP_DVECEXPR(+=)
    _DEFINE_INPLACE_OP_DVECEXPR(-=)
    _DEFINE_INPLACE_OP_DVECEXPR(=)
#undef _DEFINE_INPLACE_OP_DVECEXPR

#define _DEFINE_INPLACE_OP_SCALAR(_Op)                                   \
    template <                                                           \
        typename T,                                                      \
        typename std::enable_if<std::is_arithmetic_v<T>, int>::type = 0> \
    DVector<X, NBORDER>& operator _Op(const T& a) {                      \
        for (auto i = 0; i != _head.size(); i++)                         \
            _head[i] _Op a;                                              \
                                                                         \
        for (auto i = 0; i != _tail.size(); i++)                         \
            _tail[i] _Op a;                                              \
        return *this;                                                    \
    }
    _DEFINE_INPLACE_OP_SCALAR(*=)
    _DEFINE_INPLACE_OP_SCALAR(/=)
    _DEFINE_INPLACE_OP_SCALAR(=)
#undef _DEFINE_INPLACE_OP_SCALAR

    /***********************************************************************************/
    // Move constructors and assignement
    /***********************************************************************************/

    // Move constructor from communicator, data and tail elements
    template <typename... TAIL>
    DVector(MPI_Comm comm, X&& x, TAIL... tail)
        : _dinfo(comm)
        , _head(std::forward<X>(x))
        , _other(similar(_head)) // this assumes _head is initialised first
        , _tail({ static_cast<double>(tail)... }) {
        static_assert(sizeof...(TAIL) == NBORDER,
                      "variadic input must be consistent with the template parameter.");
    }

    /***********************************************************************************/
    // Field accessors and Indexing
    /***********************************************************************************/
    // Accessors
#define _DEFINE_ACCESSOR_OPERATOR(_Member)             \
    const auto& _Member() const { return _##_Member; } \
    auto&       _Member() { return _##_Member; }

    _DEFINE_ACCESSOR_OPERATOR(head)
    _DEFINE_ACCESSOR_OPERATOR(tail)
    _DEFINE_ACCESSOR_OPERATOR(other)
    _DEFINE_ACCESSOR_OPERATOR(dinfo)
#undef _DEFINE_ACCESSOR_OPERATOR

    // Indexing of head and tail is performed as x.head(i) and x.tail(i)
    auto&       head(const std::size_t i) { return _head[i]; }
    auto&       tail(const std::size_t i) { return _tail[i]; }
    auto&       other(const std::size_t i) { return _other[i]; }
    const auto& head(const std::size_t i) const { return _head[i]; }
    const auto& tail(const std::size_t i) const { return _tail[i]; }
    const auto& other(const std::size_t i) const { return _other[i]; }

    /***********************************************************************************/
    // MPI communication stuff
    /***********************************************************************************/

    // Broadcast the tail from the last rank with id=comm_size-1 to all procs
    void bc_tail() {
        MPI_Bcast(_tail.data(),
                  NBORDER,
                  MPI_DOUBLE,
                  _dinfo.size() - 1,
                  _dinfo.comm());
    }

    // Begin non-blocking chain shift operation across one dimensional chain of processes.
    // This should be followed by a 'shift_wait'
    void shift_init(const DMatrixBandType btype) {
        // if btype is UPWARDS is true we need to move the data 'upwards', such that
        // this process sends data to a process with rank = this_process_rank - 1
        // and received from a process with rank = this_process_rank + 1. Note
        // we need to take care of the boundaries, so we use modulo arithmetic
        int dir = btype == DMatrixBandType::UPPER ? 1 : -1;
        // FIXME: make this type generic
        MPI_Irecv(_other.data(),
                  _other.size(),
                  MPI_DOUBLE,
                  (_dinfo.rank() + dir + _dinfo.size()) % _dinfo.size(),
                  0,
                  _dinfo.comm(),
                  &_requests[0]);
        MPI_Isend(_head.data(),
                  _head.size(),
                  MPI_DOUBLE,
                  (_dinfo.rank() - dir + _dinfo.size()) % _dinfo.size(),
                  0,
                  _dinfo.comm(),
                  &_requests[1]);
    }

    // wait until all shifts have been processed
    void shift_wait() {
        MPI_Waitall(2, _requests, _statuses);
    }

    // provide the blocking version too
    void shift(const DMatrixBandType btype) {
        shift_init(btype);
        shift_wait();
    }
};

/***********************************************************************************/
// Norm and dot product
/***********************************************************************************/

// FIXME: make this working not just for double
// dot product
template <typename X, size_t NBORDER>
double operator*(const DVector<X, NBORDER>& a, const DVector<X, NBORDER>& b) {
    // compute local part of dot product
    double local = a.head() * b.head();

    // Then perform an allreduce operation so all processes
    // have the value of the dot product in memory.
    double global;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, a.dinfo().comm());

    // we need to add to the dot product the product of the tails
    // living on the last rank. We first calculate them everywhere
    // (just for simplicity) and we let the last rank broadcast the
    // value to all other processes. Finally, we sum this value to
    // the global value and return
    double prod = std::inner_product(a.tail().begin(),
                                     a.tail().end(),
                                     b.tail().begin(),
                                     0);

    MPI_Bcast(&prod, 1, MPI_DOUBLE, a.dinfo().size() - 1, a.dinfo().comm());

    return global + prod;
}

// Norm of a distributed vector
template <typename X, std::size_t NBORDER>
auto norm(const DVector<X, NBORDER>& y) {
    return std::sqrt(y * y);
}

/***********************************************************************************/
// Expression templates for DVector objects
/***********************************************************************************/

// define division and multiplication by a scalar
#define _DEFINE_MULDIV_OPERATOR(_Op, _OpName)                                                          \
                                                                                                       \
    template <typename E, typename S>                                                                  \
    class DVector##_OpName : public DVectorExpr<DVector##_OpName<E, S>> {                              \
    private:                                                                                           \
        const DVectorExpr<E>& _u;                                                                      \
        const S&              _v;                                                                      \
                                                                                                       \
    public:                                                                                            \
        DVector##_OpName(const DVectorExpr<E>& u, const S& v)                                          \
            : _u(u)                                                                                    \
            , _v(v) {}                                                                                 \
                                                                                                       \
        auto& dinfo() const { return _u.dinfo(); }                                                     \
        auto& head() const { return _u.head(); }                                                       \
        auto& tail() const { return _u.tail(); }                                                       \
        auto  other(const std::size_t i) const { return _u.other(i) _Op _v; }                          \
        auto  head(const std::size_t i) const { return _u.head(i) _Op _v; }                            \
        auto  tail(const std::size_t i) const { return _u.tail(i) _Op _v; }                            \
    };                                                                                                 \
                                                                                                       \
    template <typename E, typename S, typename std::enable_if<std::is_arithmetic_v<S>, int>::type = 0> \
    DVector##_OpName<E, S> operator _Op(const DVectorExpr<E>& u, const S& v) {                         \
        return { u, v };                                                                               \
    }                                                                                                  \
                                                                                                       \
    template <typename E, typename S, typename std::enable_if<std::is_arithmetic_v<S>, int>::type = 0> \
    DVector##_OpName<E, S> operator _Op(const S& v, const DVectorExpr<E>& u) {                         \
        return { u, v };                                                                               \
    }

_DEFINE_MULDIV_OPERATOR(*, Mul)
_DEFINE_MULDIV_OPERATOR(/, Div)
#undef _DEFINE_MULDIV_OPERATOR

// define linear space operations + and -
#define _DEFINE_ADDSUB_OPERATOR(_Op, _OpName)                                          \
                                                                                       \
    template <typename E1, typename E2>                                                \
    class DVector##_OpName : public DVectorExpr<DVector##_OpName<E1, E2>> {            \
    private:                                                                           \
        const DVectorExpr<E1>& _u;                                                     \
        const DVectorExpr<E2>& _v;                                                     \
                                                                                       \
    public:                                                                            \
        DVector##_OpName(const DVectorExpr<E1>& u,                                     \
                         const DVectorExpr<E2>& v)                                     \
            : _u(u)                                                                    \
            , _v(v) {}                                                                 \
        auto& dinfo() const { return _u.dinfo(); }                                     \
        auto& head() const { return _u.head(); }                                       \
        auto& tail() const { return _u.tail(); }                                       \
        auto  head(const std::size_t i) const { return _u.head(i) _Op _v.head(i); }    \
        auto  other(const std::size_t i) const { return _u.other(i) _Op _v.other(i); } \
        auto  tail(const std::size_t i) const { return _u.tail(i) _Op _v.tail(i); }    \
    };                                                                                 \
                                                                                       \
    template <typename E1, typename E2>                                                \
    DVector##_OpName<E1, E2>                                                           \
    operator _Op(const DVectorExpr<E1>& u, const DVectorExpr<E2>& v) {                 \
        return { u, v };                                                               \
    }

_DEFINE_ADDSUB_OPERATOR(+, Add)
_DEFINE_ADDSUB_OPERATOR(-, Sub)
#undef _DEFINE_ADDSUB_OPERATOR

} // namespace PaKr