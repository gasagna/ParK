#pragma once
#include <cassert>
#include <cmath>
#include <initializer_list>
#include <type_traits>

namespace Test {

// forward declarations
template <typename T>
class SqMat;

template <typename T>
struct VecSimilar;

// matvec product object
template <typename M, typename V>
class SqMatVec {
private:
    M const& _mat;
    V const& _vec;

public:
    SqMatVec(M const& mat, V const& vec)
        : _mat(mat)
        , _vec(vec) {
        assert(vec.size() == mat.size(1));
    }

    int size() const {
        return _vec.size();
    }

    void execute(V& x) const {
        // check sizes
        assert(x.size() == _vec.size());

        // use this for partial sums
        auto val = _mat(0, 0) * _vec(0);

        // naive matmul
        for (auto i = 0; i != x.size(); i++) {
            val = 0;
            for (auto j = 0; j != x.size(); j++) {
                val += _mat(i, j) * _vec(j);
            }
            x[i] = val;
        }
    }
};

// base class for expression template
template <typename E, typename T>
class VecExpr {
public:
    T operator[](int i) const {
        return static_cast<const E&>(*this)[i];
    }

    std::size_t size() const {
        return static_cast<const E&>(*this).size();
    }
};

// We define a custom class that implements the required interface by PaKr
template <typename T>
class Vec : public VecExpr<Vec<T>, T> {
private:
    std::vector<T> _vec;

public:
    // default constructor
    Vec() {}

    // from a size
    Vec(const int n)
        : _vec(n) {}

    // from data (makes a copy)
    Vec(std::vector<T>& data)
        : _vec(data) {
    }

    // from an initilizer list
    Vec(std::initializer_list<T> init)
        : _vec(init) {
    }

    // move constructor from another Vec
    Vec(Vec<T>&& a)
        : _vec(std::forward<std::vector<T>>(a.vec())) {
        }

    // construct from matvec object
    Vec(const SqMatVec<SqMat<T>, Vec<T>>& mv) {
        _vec.resize(mv.size());
        mv.execute(*this);
    }

    // copy constructor from a VecSimilar
    Vec(VecSimilar<T>&& sim)
        : _vec(sim.size) {}

    // copy constructor from a VecExpr
    template <typename E>
    Vec(const VecExpr<E, T>& a) {
        _vec.resize(a.size());
        for (auto i = 0; i != a.size(); i++)
            _vec[i] = a[i];
    }

    // copy constructor from a VecExpr
    Vec(const Vec<T>& a) {
        _vec.resize(a.size());
        for (auto i = 0; i != a.size(); i++)
            _vec[i] = a[i];
    }

    // assign from matvec object
    Vec<T>& operator=(const SqMatVec<SqMat<T>, Vec<T>>& mv) {
        mv.execute(*this);
        return *this;
    }
    
    // assign from an initilizer list
    Vec<T>& operator=(std::initializer_list<T> init) {
        _vec = init;
        return *this;
    }
    
    // equality operator
    bool operator==(Vec<T> const& other) const {
        if (other.size() != this->size())
            return false;
        for (auto i = 0; i != _vec.size(); i++)
            if (_vec[i] != other[i])
                return false;
        return true;
    }

// define all inplace operations +=, -=, *=, /= and also =, first with
// a VecExpr as an input and then with numbers
#define _DEFINE_INPLACE_OP(_Op)                                          \
                                                                         \
    template <typename E>                                                \
    Vec& operator _Op##=(const VecExpr<E, T>& a) {                       \
        assert(a.size() == this->size());                                \
        for (auto i = 0; i != _vec.size(); i++)                          \
            _vec[i] _Op## = a[i];                                        \
        return *this;                                                    \
    }                                                                    \
                                                                         \
    template <                                                           \
        typename S,                                                      \
        typename std::enable_if<std::is_arithmetic_v<S>, int>::type = 0> \
    Vec& operator _Op##=(const S& val) {                                 \
        for (auto i = 0; i != _vec.size(); i++)                          \
            _vec[i] _Op## = val;                                         \
        return *this;                                                    \
    }

    _DEFINE_INPLACE_OP(+)
    _DEFINE_INPLACE_OP(-)
    _DEFINE_INPLACE_OP(*)
    _DEFINE_INPLACE_OP(/)
    _DEFINE_INPLACE_OP()

#undef _DEFINE_INPLACE_OP

    // dimension of vector
    auto size() const {
        return _vec.size();
    }

    // return the underlying data
    const std::vector<T>& vec() const { return _vec; }
    std::vector<T>&       vec() { return _vec; }

    // return a raw pointer to the underlying data (for MPI)
    const T* data() const { return _vec.data(); }
    T*       data() { return _vec.data(); }

    // provide [] and () for indexing
    const T& operator()(int i) const { return _vec[i]; }
    T&       operator()(int i) { return _vec[i]; }
    const T& operator[](int i) const { return _vec[i]; }
    T&       operator[](int i) { return _vec[i]; }
};

// define dot product and norm
template <typename T>
double operator*(const Vec<T>& a, const Vec<T>& b) {
    assert(a.size() == b.size());
    double sum = 0;
    for (auto i = 0; i != a.size(); i++)
        sum += a[i] * b[i];
    return sum;
}

template <typename T>
double norm(const Vec<T>& a) {
    return std::sqrt(a * a);
}

// Struct to hold parameters for an efficient constructor that does not
// perform copies, but simply initialises the storage
template <typename T>
struct VecSimilar {
    size_t size;
};

template <typename T>
VecSimilar<T> similar(const Vec<T>& vec) {
    return { vec.size() };
}

template <typename T>
Vec<T> materialise(VecSimilar<T>&& vecsim) {
    return Vec<T>(std::forward<VecSimilar<T>>(vecsim));
}

// define division and multiplication by a scalar
#define _DEFINE_MULDIV_OPERATOR(_Op, _VecOp)                             \
                                                                         \
    template <typename E, typename T, typename S>                        \
    class _VecOp : public VecExpr<_VecOp<E, T, S>, T> {                  \
    private:                                                             \
        VecExpr<E, T> const& _u;                                         \
        S const&             _v;                                         \
                                                                         \
    public:                                                              \
        _VecOp(VecExpr<E, T> const& u, S const& v)                       \
            : _u(u)                                                      \
            , _v(v) {                                                    \
        }                                                                \
        T      operator[](size_t i) const { return _u[i] _Op _v; }       \
        size_t size() const { return _u.size(); }                        \
    };                                                                   \
                                                                         \
    template <                                                           \
        typename E,                                                      \
        typename T,                                                      \
        typename S,                                                      \
        typename std::enable_if<std::is_arithmetic_v<S>, int>::type = 0> \
    _VecOp<E, T, S> operator _Op(VecExpr<E, T> const& u, S const& v) {   \
        return { u, v };                                                 \
    }                                                                    \
                                                                         \
    template <                                                           \
        typename E,                                                      \
        typename T,                                                      \
        typename S,                                                      \
        typename std::enable_if<std::is_arithmetic_v<S>, int>::type = 0> \
    _VecOp<E, T, S> operator _Op(S const& v, VecExpr<E, T> const& u) {   \
        return { u, v };                                                 \
    }

_DEFINE_MULDIV_OPERATOR(*, VecMul)
_DEFINE_MULDIV_OPERATOR(/, VecDiv)
#undef _DEFINE_MULDIV_OPERATOR

// define linear space operations + and -
#define _DEFINE_ADDSUB_OPERATOR(_Op, _VecOp)                                           \
                                                                                       \
    template <typename E1, typename E2, typename T>                                    \
    class _VecOp : public VecExpr<_VecOp<E1, E2, T>, T> {                              \
    private:                                                                           \
        VecExpr<E1, T> const& _u;                                                      \
        VecExpr<E2, T> const& _v;                                                      \
                                                                                       \
    public:                                                                            \
        _VecOp(VecExpr<E1, T> const& u, VecExpr<E2, T> const& v)                       \
            : _u(u)                                                                    \
            , _v(v) {                                                                  \
            assert(u.size() == v.size());                                              \
        }                                                                              \
                                                                                       \
        T      operator[](size_t i) const { return _u[i] _Op _v[i]; }                  \
        size_t size() const { return _v.size(); }                                      \
    };                                                                                 \
                                                                                       \
    template <typename E1, typename E2, typename T>                                    \
    _VecOp<E1, E2, T> operator _Op(VecExpr<E1, T> const& u, VecExpr<E2, T> const& v) { \
        return { u, v };                                                               \
    }

_DEFINE_ADDSUB_OPERATOR(+, VecAdd)
_DEFINE_ADDSUB_OPERATOR(-, VecSub)
#undef _DEFINE_ADDSUB_OPERATOR

// example square matrix class
template <typename T>
class SqMat {
private:
    std::vector<T> _data;
    int            _n;

public:
    // from an initilizer list
    SqMat(std::initializer_list<T> init, int n)
        : _data(init)
        , _n(n) {
        assert(_data.size() == n * n);
    }

    // from the size and a value
    SqMat(std::size_t n, T val = T())
        : _data(n * n, val)
        , _n(n) {
    }

    // this is a square matrix!
    int size(int) const {
        return _n;
    }

    // indexing
    const T& operator()(int i, int j) const { return _data[i + j * _n]; }
    T&       operator()(int i, int j) { return _data[i + j * _n]; }

    // return matvec object for product
    SqMatVec<SqMat<T>, Vec<T>> operator*(Vec<T> const& v) const {
        return { *this, v };
    }
};

} // namespace Test