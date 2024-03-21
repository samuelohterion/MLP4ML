#ifndef ALGEBRA_HPP
#define ALGEBRA_HPP

#include <algorithm>
#include <bitset>
#include <complex>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <math.h>
#include <ostream>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <string.h>
#include <type_traits>
#include <vector>

namespace alg {

	class STR :
	public std::string {

		public:

			STR() : std::string() {

			}

			STR(std::string const & p_s) : std::string(p_s) {

			}

			STR(char const * p_s) : std::string(p_s) {

			}

			STR(int const & p_number) : std::string() {

				std::stringstream
				ss;

				ss << p_number;

				*this = ss.str();
			}

			STR
			operator + () const {

				return std::operator + ("+", *this);
			}

			STR
			operator - () const {

				return std::operator + ("-", *this);
			}

			STR
			operator + (STR const & p_s) const {

				return std::operator + (std::operator + (std::operator + (std::operator + ("(", *this), " + "), p_s), ")");
			}

			STR
			operator - (STR const & p_s) const {

				return std::operator + (std::operator + (std::operator + (std::operator + ("(", *this), " - "), p_s), ")");
			}

			STR
			operator * (STR const & p_s) const {

				return std::operator + (std::operator + (*this, " * "), p_s);
			}

			STR
			operator / (STR const & p_s) const {

				return std::operator + (std::operator + (*this, " / "), p_s);
			}


			STR
			operator += (STR const & p_s) {

				return *this = *this + p_s;
			}

			STR
			operator -= (STR const & p_s) {

				return *this = *this - p_s;
			}

			STR
			operator *= (STR const & p_s) {

				return *this = *this * p_s;
			}

			STR
			operator /= (STR const & p_s) {

				return *this = *this / p_s;
			}
	};

	typedef std::size_t SIZE;

	template < typename SOME_INTEGER_TYPE >
	bool
	gbit(SOME_INTEGER_TYPE const & p_bitset, SIZE const & p_bitId) {

		return (p_bitset >> p_bitId) & 1;
	}	

	bool
	gbit( char * p_bitset, SIZE const & p_bitId ) {

		SIZE
		byteId = p_bitId >> 3,
		bitId  = p_bitId & 0x7;

		return ( p_bitset[ byteId ] >> bitId ) & 1;
	}

	void
	cbit( char * p_bitset, SIZE const & p_bitId ) {

		SIZE
		byteId = p_bitId >> 3,
		bitId  = p_bitId & 0x7;

		p_bitset[ byteId ] &= static_cast<char>( ~( 1 << bitId ) );
	}

	template < typename SOME_INTEGER_TYPE >
	void
	cbit(SOME_INTEGER_TYPE  & p_bitset, SIZE const & p_bitId) {

		p_bitset &= (~(1 << p_bitId));
	}

	template < typename SOME_INTEGER_TYPE >
	void
	sbit(SOME_INTEGER_TYPE  & p_bitset, SIZE const & p_bitId) {

		p_bitset |= (1 << p_bitId);
	}

	void
	sbit( char * p_bitset, SIZE const & p_bitId ) {

		SIZE
		byteId = p_bitId >> 3,
		bitId  = p_bitId & 0x7;

		p_bitset[ byteId ] |= static_cast<char>( 1 << bitId );
	}
	
	// vector
	template < typename T > using
	Vec = std::vector< T >;

	// matrix
	template < typename T > using
	Mat = std::vector< std::vector< T > >;

	// tensor 3rd degree
	template < typename T > using
	Tsr = std::vector< std::vector< std::vector< T > > >;

	// tensor 4rd degree
	template < typename T > using
	Tsr4 = std::vector< std::vector< std::vector< std::vector< T > > > >;

	// unsigned indices
	typedef Vec< SIZE > UIDX;
	typedef UIDX IDX ;

	// signed indices
	typedef Vec< long > SIDX ;


	template < typename T >
	inline SIZE
	len(Vec< T > const & p_vec) {

		return p_vec.size();
	}

	template < typename T >
	inline SIZE
	nrows(Mat< T > const & p_mat) {

		return p_mat.size();
	}

	template < typename T >
	inline SIZE
	ncols(Mat< T > const & p_mat) {

		return p_mat[0].size();
	}

	template < typename T, class Fun >
	inline void
	fOr(Vec< T > const & p_vec, Fun const & p_fun) {

		std::for_each(p_vec.begin(), p_vec.end(), p_fun);
	}

	template < typename T, class Fun >
	inline  Vec< T >
	trnsfrm(Vec< T > const & p_vec, Fun const & p_fun) {

		Vec< T >
		r(p_vec.size());

		std::transform(p_vec.cbegin(), p_vec.cend(), r.begin(), p_fun);

		return r;
	}

	template < typename T, class Fun >
	inline Vec< T >
	trnsfrm(Vec< T > const & p_vec1, Vec< T > const & p_vec2, Fun const & p_fun) {

		Vec< T >
		r(p_vec1.size());

		std::transform(p_vec1.cbegin(), p_vec1.cend(), p_vec2.cbegin(), r.begin(), p_fun);

		return r;
	}

	//template< typename T = double > inline T _plus(T const & p_v1, T const & p_v2) {return p_v1 + p_v2;}
	//template< typename T = double > inline T _minus(T const & p_v1, T const & p_v2) {return p_v1 - p_v2;}
	//template< typename T = double > inline T _multiplies(T const & p_v1, T const & p_v2) {return p_v1 * p_v2;}
	//template< typename T = double > inline T _divides(T const & p_v1, T const & p_v2) {return p_v1 / p_v2;}

#define BINOPVV(OP, foo) \
	template< typename T = double > inline Vec< T > operator OP(Vec< T > const & p_lhs, Vec< T > const & p_rhs) {Vec< T > ret(p_lhs.size()); \
	std::transform(p_lhs.cbegin(), p_lhs.cend(), p_rhs.cbegin(), ret.begin(), foo()); return ret;}

	BINOPVV(+, std::plus< T >)
	BINOPVV(-, std::minus< T >)
	BINOPVV(*, std::multiplies< T >)
	BINOPVV(/, std::divides< T >)

#define BINASSGNOPVV(OP) \
	template< typename T = double > inline Vec< T > & operator OP(Vec< T > & p_lhs, Vec< T > const & p_rhs) {\
	auto il = p_lhs.begin(), el = p_lhs.end(); auto ir = p_rhs.cbegin(); while(il != el) {*il OP *ir; ++il; ++ir;} return p_lhs;}

	BINASSGNOPVV(+=)
	BINASSGNOPVV(-=)
	BINASSGNOPVV(*=)
	BINASSGNOPVV(/=)

#define BINOPSV(OP) \
	template< typename T = double > inline Vec< T > operator OP(T const & p_lhs, Vec< T > p_rhs) {for(T & v : p_rhs) v = p_lhs OP v; return p_rhs;}

	BINOPSV(+)
	BINOPSV(-)
	BINOPSV(*)
	BINOPSV(/)

#define BINOPVS(OP) \
	template< typename T = double > inline Vec< T > operator OP(Vec< T > p_lhs, T const & p_rhs) {for(T & v : p_lhs) v = v OP p_rhs; return p_lhs;}

	BINOPVS(+)
	BINOPVS(-)
	BINOPVS(*)
	BINOPVS(/)

#define BINASSGNOPVS(OP) \
	template< typename T = double > inline Vec< T > & operator OP(Vec< T > & p_lhs, T const & p_rhs) {for(T & v : p_lhs) v OP p_rhs; return p_lhs;}

	BINASSGNOPVS(+=)
	BINASSGNOPVS(-=)
	BINASSGNOPVS(*=)
	BINASSGNOPVS(/=)

#define BINOPMM(OP) \
	template< typename T = double > inline Mat< T > operator OP(Mat< T > const & p_lhs, Mat< T > const & p_rhs) {\
	Mat< T > ret(p_lhs.size()); for(SIZE i = 0; i < ret.size(); ++ i) ret[i] = p_lhs[i] OP p_rhs[i];	return ret;}

	BINOPMM(+)
	BINOPMM(-)
	BINOPMM(*)
	BINOPMM(/)

#define BINASSGNOPMM(OP) \
	template< typename T = double > inline Mat< T > & operator OP(Mat< T > & p_lhs, Mat< T > const & p_rhs) {\
	for(SIZE i = 0; i < p_lhs.size(); ++ i) p_lhs[i] OP p_rhs[i]; return p_lhs;}

	BINASSGNOPMM(+=)
	BINASSGNOPMM(-=)
	BINASSGNOPMM(*=)
	BINASSGNOPMM(/=)

#define BINOPSM(OP) \
	template< typename T = double > inline Mat< T > operator OP(T const & p_lhs, Mat< T > const & p_rhs) {Mat< T > ret(p_rhs.size()); \
	for(SIZE i = 0; i < ret.size(); ++ i) ret[i] = p_lhs OP p_rhs[i]; return ret;}

	BINOPSM(+)
	BINOPSM(-)
	BINOPSM(*)
	BINOPSM(/)

#define BINOPMS(OP) \
	template< typename T = double > inline Mat< T > operator OP(Mat< T > const & p_lhs, T const & p_rhs) {Mat< T > ret(p_lhs.size()); \
	for(SIZE i = 0; i < p_lhs.size(); ++ i) ret[i] = p_lhs[i] OP p_rhs; return ret;}

	BINOPMS(+)
	BINOPMS(-)
	BINOPMS(*)
	BINOPMS(/)

#define BINASSGNOPMS(OP) \
	template< typename T = double > inline Mat< T > & operator OP(Mat< T > & p_lhs, T const & p_rhs) {\
	for(auto & i : p_lhs) i OP p_rhs; return p_lhs;}

	BINASSGNOPMS(+=)
	BINASSGNOPMS(-=)
	BINASSGNOPMS(*=)
	BINASSGNOPMS(/=)

#define BINOPVM(OP) \
	template< typename T = double > inline Mat< T > operator OP(Vec< T > const & p_lhs, Mat< T > const & p_rhs) {Mat< T > ret(p_rhs.size()); \
	for(SIZE i = 0; i < ret.size(); ++ i) ret[i] = p_lhs OP p_rhs[i]; return ret;}

	BINOPVM(+)
	BINOPVM(-)
	BINOPVM(*)
	BINOPVM(/)

#define BINOPMV(OP) \
	template< typename T = double > inline Mat< T > operator OP(Mat< T > const & p_lhs, Vec< T > const & p_rhs) {Mat< T > ret(p_lhs.size(), Vec< T >(p_lhs[0])); \
	for(SIZE r = 0; r < ret.size(); ++ r) for(SIZE c = 0; c < p_lhs[r].size(); ++ c) ret[r][c] = p_lhs[r][c] OP p_rhs[r]; return ret;}

	BINOPMV(+)
	BINOPMV(-)
	BINOPMV(*)
	BINOPMV(/)

#define BINASSGNOPMV(OP) \
	template< typename T = double > inline Mat< T > & operator OP(Mat< T > & p_lhs, Vec< T > const & p_rhs) {\
	SIZE rows = nrows(p_lhs), cols = ncols(p_lhs); \
	for(SIZE r = 0; r < rows; ++ r) for(SIZE c = 0; c < cols; ++ c) p_lhs[r][c] OP p_rhs[r]; return p_lhs;}

	BINASSGNOPMV(+=)
	BINASSGNOPMV(-=)
	BINASSGNOPMV(*=)
	BINASSGNOPMV(/=)

	template< typename T = double >
	inline Mat< T >
	t(Mat< T > const & p_mat) {

	Mat< T >
	ret(ncols(p_mat), Vec< T >(nrows(p_mat)));

	for(SIZE r = 0; r < nrows(ret); ++ r) {

		for(SIZE c = 0; c < ncols(ret); ++ c) {

			ret[r][c] = p_mat[c][r];
		}
	}

	return ret;
}

template< typename T = double >
inline Mat< T >
adj(Mat< T > p_mat) {

	SIZE
	rows = nrows(p_mat),
	cols = ncols(p_mat);

	for(SIZE r = 0; r < rows; ++ r) {

		for(SIZE c = 0; c < cols; ++ c) {

			p_mat[r][c] = conj(p_mat[c][r]);
		}
	}

	return p_mat;
}

template< typename T = double >
inline Mat< T >
conj(Mat< T > p_mat) {

	SIZE
	rows = ncols(p_mat),
	cols = nrows(p_mat);

	for(SIZE r = 0; r < rows; ++ r) {

		for(SIZE c = 0; c < cols; ++ c) {

			p_mat[r][c] = conj(p_mat[c][r]);
		}
	}

	return p_mat;
}

template< typename T = double >
inline Mat< T >
operator ~(Mat< T > const & p_mat) {

	return t(p_mat);
}

template< typename T = double >
inline Mat< std::complex< T > >
operator ~(Mat< std::complex< T > > const & p_mat) {

	return conj< std::complex< T > >(p_mat);
}

template< typename T = double >
inline Vec< T >
operator +(Vec< T > const & p_vec) {

	return p_vec;
}

template< typename T = double >
inline Mat< T >
operator +(Mat< T > const & p_mat) {

	return p_mat;
}

template< typename T = double >
inline Vec< T >
operator -(Vec< T > const & p_vec) {

	Vec< T >
	ret(p_vec.size());

	for(SIZE  c = 0; c < ret.size(); ++ c) {

			ret[c] = -p_vec[c];
		}

		return ret;
}

template< typename T = double >
inline Mat< T >
operator -(Mat< T > const & p_mat) {

	Mat< T >
		ret(nrows(p_mat), Vec< T >(ncols(p_mat)));

		for(SIZE r = 0; r < nrows(ret); ++ r) {

			for(SIZE c = 0; c < ncols(ret); ++ c) {

				ret[r][c] = - p_mat[r][c];
			}
		}

		return ret;
}


template< typename T = double >
inline Mat< T >
	operator ^(Vec< T > const & p_lhs, Vec< T > const & p_rhs) {

		Mat< T >
		ret;

		for(auto v : p_lhs)

			ret.push_back(v * p_rhs);

		return ret;
	}

	template< typename T = double >
	inline Vec< T >
	operator |(T const & p_lhs, Vec< T > const & p_rhs) {

		return p_lhs * p_rhs;
	}

	template< typename T = double >
	inline Vec< T >
	operator |(Vec< T > const & p_lhs, T const & p_rhs) {

		return p_lhs * p_rhs;
	}

	template< typename T = double >
	inline Vec< T >
	& operator |=(Vec< T > & p_lhs, T const & p_rhs) {

		return p_lhs *= p_rhs;
	}

	template< typename T = double >
	inline Mat< T >
	operator |(T const & p_lhs, Mat< T > const & p_rhs) {

		return p_lhs * p_rhs;
	}

	template< typename T = double >
	inline Mat< T >
	operator |(Mat< T > const & p_lhs, T const & p_rhs) {

		return p_lhs * p_rhs;
	}

	template< typename T = double >
	inline Mat< T >
	& operator |=(Mat< T > & p_lhs, T const & p_rhs) {

		return p_lhs = p_lhs | p_rhs;
	}

	template< typename T = double >
	inline T
	operator |(Vec< T > const & p_lhs, Vec< T > const & p_rhs) {

		T
		s = p_lhs[0] * p_rhs[0];

		for(SIZE i = 1; i < p_lhs.size(); ++ i )

			s +=  p_lhs[i] * p_rhs[i];

		return s;
	}

	template< typename T = double >
	inline Vec< T >
	operator |(Vec< T > const & p_lhs, Mat< T > const & p_rhs) {

		SIZE
		cols = ncols(p_rhs);

		Vec< T >
		ret(cols);

		for(SIZE c = 0; c < cols; ++ c) {

			T
			sum = p_lhs[0] * p_rhs[0][c];

			for(SIZE r = 1; r < p_lhs.size(); ++ r) {

				sum += p_lhs[r] * p_rhs[r][c];
			}

			ret[c] = sum;
		}

		return ret;
	}

	template< typename T = double >
	inline Vec< T >
	& operator |=(Vec< T > & p_lhs, Mat< T > const & p_rhs) {

		p_lhs = p_lhs | p_rhs;

		return p_lhs;
	}

	template< typename T = double >
	inline Vec< T >
	operator |(Mat< T > const & p_lhs, Vec< T > const & p_rhs) {

		SIZE
		rows = nrows(p_lhs);

		Vec< T >
		ret(rows);

		for(SIZE r = 0; r < rows; ++ r) {

			ret[r] = p_lhs[r] | p_rhs;
		}

		return ret;
	}

	template< typename T = double >
	inline Mat< T >
	operator |(Mat< T > const & p_lhs, Mat< T > const & p_rhs) {

		SIZE
		rows = nrows(p_lhs),
		cols = ncols(p_rhs),
		size = nrows(p_rhs);

		Mat< T >
		ret(rows, Vec< T >(cols));

		for(SIZE r = 0; r < rows; ++ r) {

			for(SIZE c = 0; c < cols; ++ c) {

				T
				s = p_lhs[r][0] * p_rhs[0][c];

				for(SIZE i = 1; i < size; ++ i) {

					s += p_lhs[r][i] * p_rhs[i][c];
				}

				ret[r][c] =  s;
			}
		}

		return ret;
	}

	template< typename T = double >
	inline Mat< T >
	& operator |=(Mat< T > & p_lhs, Mat< T > const & p_rhs) {

		p_lhs = p_lhs | p_rhs;

		return p_lhs;
	}

	template< typename T = double >
	inline T
	norm(Vec< T > const & p_vec) {

		return sqrt(p_vec | p_vec);
	}

	template< typename T = double >
	inline Mat< T >
	eye(SIZE const & p_rows) {

		Mat< T >
		ret(p_rows, Vec< T >(p_rows));

		for(SIZE r = 0; r < nrows(ret); ++ r)

			for(SIZE c = 0; c < ncols(ret); ++ c)

				ret[r][c] = r == c ? T(1) : T(0);

		return ret;
	}

	template< typename T = double >
	inline Vec< T >
	vcnst(SIZE const & p_rows = 1, T const & p_const = 0) {

		return Vec< T >(p_rows, p_const);
	}

	template< typename T = double >
	inline Mat< T >
	mcnst(SIZE const & p_rows = 1, SIZE const & p_cols = 0, T const & p_const = 0) {

		return Mat< T >(p_rows, Vec< T >(p_cols == 0 ? p_rows : p_cols, p_const));
	}

	template< typename T = double >
	inline Tsr< T >
	tcnst(SIZE const & p_cells = 1, SIZE const & p_rows = 0, SIZE const & p_cols = 0, T const & p_const = 0) {

		return Tsr< T >(p_cells, Mat< T >(p_rows == 0 ? p_cells : p_rows, p_cols, p_const));
	}

	template< typename T = double >
	inline Vec< T >
	vrnd(SIZE const & p_size) {

		Vec< T >
		ret(p_size);

		double rand_max_rec = 1. / RAND_MAX;

		for(auto & r : ret)

			r = static_cast< double >(std::rand()) * rand_max_rec;

		return ret;
	}

	template< typename T = double >
	inline Mat< T >
	mrnd(SIZE const & p_rows, SIZE const & p_cols = 0) {

		Mat< T >
		ret(p_rows);

		for(auto & r : ret)

			r = vrnd< T >(p_cols < 1 ? p_rows : p_cols);

		return ret;
	}

	template< typename T = double >
	inline Tsr< T >
	trnd(SIZE const & p_cells, SIZE const & p_rows = 0, SIZE const & p_cols = 0) {

		Tsr< T >
		ret(p_cells);

		for(auto & r : ret)

			r = mrnd< T >(p_rows < 1 ? p_cells : p_rows, p_cols);

		return ret;
	}

	template< typename T = double >
	inline SIZE
	maxStringLengthOfElements(Vec< T > const & p_vec) {

		SIZE
		len = 0,
		llen = 0;

		std::stringstream
		ss;

		for(auto v : p_vec) {

			ss.str("");

			ss << v;

			llen = ss.str().length();

			if(len < llen)

				len = llen;
		}

		return len;
	}

	template< typename T = double >
	inline std::string
	vec2Str(Vec< T > const & p_vec, int const & p_width = 0) {

		std::stringstream
		ss;

		for(auto v : p_vec) {

			if(0 < p_width)

				ss << std::setw(p_width);

			ss << v;
		}

		return ss.str();
	}

	template< typename T = double >
	inline std::ostream
	& operator << (std::ostream & p_os, Vec< T > const & p_vec) {

		for(auto v : p_vec)

			p_os << v << "  ";

		return p_os;
	}

	template< typename T = double >
	inline std::ostream
	& operator << (std::ostream & p_os, Mat< T > const & p_mat) {

		SIZE
		len = 0,
		llen = 0;

		for(auto v : p_mat) {

			llen = maxStringLengthOfElements(v);

			if(len < llen)

				len = llen;
		}

		auto
		v = p_mat.cbegin();

		while(v < p_mat.cend() - 1)

			p_os << vec2Str(*v++, static_cast<int>(len + 1)) << std::endl;

		p_os << vec2Str(*v, static_cast<int>(len + 1));

		return p_os;
	}

	template< typename T = double >
	inline std::ostream
	& operator << (std::ostream & p_os, Tsr< T > const & p_ten) {

		for(auto v : p_ten)

			p_os << v << std::endl << std::endl;

		return p_os;
	}

	template< typename T = double >
	inline std::ostream
	& operator << (std::ostream & p_os, Tsr4< T > const & p_ten) {

		for(auto v : p_ten)

			p_os << v << std::endl;

		return p_os;
	}

	template< typename T = double >
	inline T
	sum(Vec< T > const & p_v) {

		T s = 0;

		for(auto x : p_v) s += x;

		return s;
	}

	template< typename T = double >
	inline T
	sum(Mat< T > const & p_m) {

		T s = 0;

		for(auto x : p_m) s += sum(x);

		return s;
	}

	template< typename T = double >
	inline T
	sum(Tsr< T > const & p_t) {

		T s = 0;

		for(auto x : p_t) s += sum(x);

		return s;
	}

	inline void
	print(std::string const & p_string) {

		std::cout << p_string << std::endl;
	}

	template< typename T = double >
	inline void
	print(std::string const & p_string, T const & p_v) {

		print(p_string);

		std::cout << p_v << std::endl << std::endl;
	}

	inline SIZE
	idxOfNxt(SIZE const & p_val, SIZE p_idx, Vec< SIZE > const & p_vec) {

		while(p_idx < p_vec.size()) {

			if(p_vec[p_idx] == p_val) {

				return p_idx;
			}

			++ p_idx;
		}

		return p_idx;
	}

	inline SIZE
	idxOf1st(SIZE const & p_val, Vec< SIZE > const & p_vec) {

		return idxOfNxt(p_val, 0, p_vec);
	}

	inline Vec< SIZE >
	permutationZero(SIZE const & p_size) {

		Vec< SIZE >
		permutation(p_size);

		for(SIZE i = 0; i < p_size; ++ i) {

			permutation[i] = i;
		}

		return permutation;
	}

	inline Vec< SIZE >
	& nextPermutation(Vec< SIZE > & p_permutation) {

		SIZE
		curr = p_permutation.size();

		while(0 < curr --) {

			SIZE
			greatest = idxOf1st(curr, p_permutation),
			nextHole = idxOfNxt(p_permutation.size(), greatest + 1, p_permutation);

			if(nextHole < p_permutation.size()) {

				p_permutation[nextHole] = curr;
				p_permutation[greatest] = p_permutation.size();

				nextHole = idxOfNxt(p_permutation.size(), 0, p_permutation);

				while(nextHole < p_permutation.size()) {

					p_permutation[nextHole] = ++ curr ;

					nextHole = idxOfNxt(p_permutation.size(), 0, p_permutation);
				}

				return p_permutation;
			}

			p_permutation[greatest] = p_permutation.size();
		}

		for(SIZE i = 0; i < p_permutation.size(); ++ i) {

			p_permutation[i] = i;
		}

		return p_permutation;
	}

	inline SIZE
	fac(SIZE p_n) {

		return p_n == 0 ? 1 : p_n * fac(p_n - 1);
	}

	inline SIZE
	findID(unsigned long long const & p_idx, SIZE p_from = 0) {

		while (p_from < (sizeof (unsigned long long) << 3)) {

			if (! gbit<unsigned long long>(p_idx, p_from))

				return p_from;

			++ p_from;
		}

		return p_from;
	}

	template< typename T = double >
	inline T
	detRec(Mat< T > const & p_m, SIZE const & p_row, unsigned long long & p_cidx) {

		SIZE
		rows = p_m.size();

		if(rows < p_row + 3) {

			SIZE
			leftColumnId = findID(p_cidx),
			rightColumnId = findID(p_cidx, leftColumnId + 1);

			return p_m[p_row][leftColumnId] * p_m[p_row + 1][rightColumnId] - p_m[p_row + 1][leftColumnId] * p_m[p_row][rightColumnId];
		}

		SIZE
		columnId = findID(p_cidx);

		T
		sgn = T(1),
		val = T(0);

		for(SIZE col = 0; col < rows - p_row; ++ col) {

			sbit(p_cidx, columnId);

			val += p_m[p_row][columnId] * sgn * detRec(p_m, p_row + 1, p_cidx);

			cbit(p_cidx, columnId);

			columnId = findID(p_cidx, columnId + 1);

			if (rows <= columnId)

				break;

			sgn = -sgn;
		}

		return val;
	}

	template< typename T = double >
	inline T
	detRec(Mat< T > const & p_m) {

		SIZE
		cols = ncols (p_m);

		if (cols != nrows (p_m)) {

			return T(0);
		}

		unsigned long long
		idx = 0ull;

		return detRec(p_m, 0, idx);
	}

	template< typename T = double >
	inline T
	detPerm(Mat< T > const & p_m, Vec< SIZE > & p_ridx, Vec< SIZE > & p_cidx) {

		if(nrows(p_m) != ncols(p_m))

			return T(0);

		T
		d = T(0);

		SIZE
		loops = fac(p_cidx.size());

		SIZE
		s01 = 0;

		Vec< SIZE >
		idx = permutationZero(p_cidx.size());

		for(SIZE i = 0; i < loops; ++ i) {

			T
			p = T(1);

			for(SIZE j = 0; j < p_cidx.size(); ++ j) {

				p *= p_m[p_ridx[j]][p_cidx[idx[j]]];
			}

			d += (((++ s01) & 0x2) == 0x2 ? -1 : 1) * p;

			idx = nextPermutation(idx);
		}

		return d;
	}

	template< typename T = double >
	inline T
	detPerm(Mat< T > const & p_m) {

		Vec< SIZE >
		cidx = permutationZero(ncols(p_m)),
		ridx = permutationZero(nrows(p_m));

		return detPerm(p_m, ridx, cidx);
	}

	template < typename T >
	class LRData {

		public:

			LRData(Mat< T > const & p_m) :
			size(nrows(p_m) == ncols(p_m) ? nrows(p_m) : 0),
			mat(size ? p_m : Mat< T >()),
			idx(size),
			sign4Det(1) {

				for(SIZE i = 0; i < len(idx); ++ i) {

					idx[i] = i;
				}
			}

		SIZE
		size;

		Mat< T >
		mat;

		IDX
		idx;

		char
		sign4Det;
	};

	template< typename T = double >
	inline LRData< T >
	decompLR(Mat< T > const & p_m) {

		LRData< T >
		lrdata(p_m);

		SIZE
		size = len(lrdata.idx);

		if(lrdata.idx.size() < 1) {

			return lrdata;
		}

		for(SIZE col = 0; col < size - 1; ++ col) {

			SIZE
			maxValRow = col;

			for(SIZE row = col + 1; row < size; ++ row) {

				if(fabs(lrdata.mat[maxValRow][col]) < fabs(lrdata.mat[row][col])) {

					maxValRow = row;
				}
			}

			if(maxValRow > col) {

				Vec< T >
				tmpV                  = lrdata.mat[col];
				lrdata.mat[col]       = lrdata.mat[maxValRow];
				lrdata.mat[maxValRow] = tmpV;

				SIZE
				tmpI                  = lrdata.idx[col];
				lrdata.idx[col]       = lrdata.idx[maxValRow];
				lrdata.idx[maxValRow] = tmpI;

				lrdata.sign4Det = -lrdata.sign4Det;
			}

			T
			factor = lrdata.mat[col][col];

			if(fabs(factor) > 0.) {

				factor = T(1) / factor;

				for(SIZE row = col + 1; row < size; ++ row) {

					T
					f = factor * lrdata.mat[row][col];

					lrdata.mat[row][col] = f;

					for(SIZE colR = col + 1; colR < size; ++ colR) {

						lrdata.mat[row][colR] -= f * lrdata.mat[col][colR];
					}
				}
			}
		}

		return lrdata;
	}

	template< typename T = double >
	inline Vec< T >
	solve(LRData< T > const & p_lr, Vec< T > const & p_v) {

		Vec< T >
		z = Vec< T >(p_lr.size),
		y = Vec< T >(p_lr.size),
		x = Vec< T >(p_lr.size);

		for(SIZE i = 0; i < p_lr.size; ++ i) {

			z[i] = p_v[p_lr.idx[i]];
		}

		for(SIZE i = 0; i < p_lr.size; ++ i) {

			T
			s = 0;

			for(SIZE j = 0; j < i; ++ j) {

				s += y[j] * p_lr.mat[i][j];
			}

			y[i] = z[i] - s;
		}

		for(SIZE i = p_lr.size - 1; i < p_lr.size; -- i) {

			T
			s = 0;

			for(SIZE j = i + 1; j < p_lr.size; ++ j) {

				s += x[j] * p_lr.mat[i][j];
			}

			x[i] = (y[i] - s)  / p_lr.mat[i][i];
		}

		return x;
	}

	template< typename T = double >
	inline Mat< T >
	solve(LRData< T > const & p_lr, Mat< T > const & p_m) {

		Mat< T >
		m = mcnst< T >(p_lr.size, p_lr.size);

		for(SIZE r = 0; r < p_lr.size; ++r) {

			m[r] = solve(p_lr, p_m[r]);
		}

		return m;
	}

	template< typename T = double >
	inline T
	det(Mat< T > const & p_m) {

		LRData< T >
		lrdata = decompLR(p_m);

		T
		det = T(1);

		for(SIZE i = 0; i < lrdata.size; ++i) {

			det *= lrdata.mat[i][i];
		}

		return lrdata.sign4Det * det;
	}

	template< typename T = double >
	inline Mat< T >
	inv(Mat< T > const & p_m) {

		return t(solve(decompLR(p_m), eye< T >(len(p_m))));
	}

	template < typename T >
	T
	round(T const & p_v, int const & p_digits = 0) {

		T
		r = exp10(-p_digits) * std::round(exp10(p_digits) * p_v);

		if(0 >= r * r) {

			r = T(0);
		}

		return r;
	}

	template < template< class > class Complex, class T >
	Complex< T >
	round(Complex< T > const & p_v, int const & p_digits = 0) {

		long double
		a = exp10l(-p_digits),
		b = exp10l(+p_digits);

		T
		r = a * std::round(b * p_v.real()),
		i = a * std::round(b * p_v.imag());

		if(0 >= r * r) r = T(0);
		if(0 >= i * i) i = T(0);

		return Complex< T >(r, i );
	}

	template < typename T >
	Vec< T >
	round(Vec< T > const & p_v, int const & p_digits = 0) {

		Vec< T >
		v = p_v;

		for(auto & s : v) {

			s = round(s, p_digits);
		}

		return v;
	}

	template < typename T >
	Mat< T >
	round(Mat< T > const & p_m, int const & p_digits = 0) {

		Mat< T >
		m = p_m;

		for(auto & s : m) {

			s = round(s, p_digits);
		}

		return m;
	}

	template < typename T >
	inline Vec< T >
	sub (Vec< T > const & p_vec, int const & p_begin, int const & p_size) {

		Vec< T >
		r;

		for(int i = p_begin; i < p_begin + p_size; ++ i) {

			r.push_back(p_vec[i]);
		}

		return r;
	}

	template < typename T >
	inline Vec< T >
	sub (Vec< T > const & p_vec, IDX const & p_idx) {

		Vec< T >
		v(len(p_idx));

		for(SIZE i = 0; i < len(p_idx); ++ i) {

			v[i] = p_vec[p_idx[i]];
		}

		return v;
	}

	template < typename T >
	inline Mat< T >
	sub (Mat< T > const & p_mat, int const & p_row, int const & p_col, int const & p_rows, int const & p_cols) {

		Mat< T >
		d;

		for(int i = p_row; i < p_row + p_rows; ++ i) {

			d.push_back(sub(p_mat[i], p_col, p_cols));
		}

		return d;
	}

	template < typename T >
	inline Mat< T >
	sub (Mat< T > const & p_mat, IDX const & p_ridx, IDX const & p_cidx) {

		Mat< T >
		m(len(p_ridx), len(p_cidx));

		for(SIZE r = 0; r < len(p_ridx); ++ r) {

			for(SIZE c = 0; c < len(p_cidx); ++ c) {

				m[r][c] = p_mat[p_ridx[r]][p_cidx[c]];
			}
		}

		return m;
	}

	template< typename T = double >
	bool
    save(std::string const & p_filename, Vec< T > const & p_vec) {

		std::ofstream
        ofs(p_filename);

		if(! ofs.is_open())

			return false;

		ofs << p_vec;

		ofs.close();

		return true;
	}

	template< typename T = double >
	bool
	load(std::ifstream & p_ifs, Vec< T > & p_vec) {

		bool
		ret = false;

		p_vec.resize(0);

		std::string
		line;

		std::getline(p_ifs, line);

		std::stringstream
		ss(line);

		T
		val;

		while(ss >> val) {

			p_vec.push_back(val);

			ret = true;
		}

		return ret;
	}

	template< typename T = double >
	bool
	load(std::ifstream & p_ifs, Mat< T > & p_mat) {

		bool
		ret = false;

		p_mat.resize(0);

		Vec< T >
		v;

		while(load(p_ifs, v)) {

			p_mat.push_back(v);

			ret = true;
		}

		return ret;
	}

	template< typename T = double >
	bool
	load(std::ifstream & p_ifs, Tsr< T > & p_ten) {

		bool
		ret = false;

		p_ten.resize(0);

		Mat< T >
		v;

		while(load(p_ifs, v)) {

			p_ten.push_back(v);

			ret = true;
		}

		return ret;
	}

	template< typename T = double >
	bool
	load(std::ifstream & p_ifs, Tsr4< T > & p_ten) {

		bool
		ret = false;

		p_ten.resize(0);

		Tsr< T >
		v;

		while(load(p_ifs, v)) {

			p_ten.push_back(v);

			ret = true;
		}

		return ret;
	}

	template< typename T = double >
	bool
	load(std::string const & p_filename, Vec< T > & p_vec) {

		std::ifstream
		ifs(p_filename);

		if(! ifs.is_open())

			return false;

		bool
		ret = load(ifs, p_vec);

		ifs.close();

		return ret;
	}

	template< typename T = double >
	bool
	load(std::string const & p_filename, Mat< T > & p_mat) {

		std::ifstream
		ifs(p_filename);

		if(! ifs.is_open())

			return false;

		bool
		ret = load(ifs, p_mat);

		ifs.close();

		return ret;
	}

	template< typename T = double >
	bool
	load(std::string const & p_filename, Tsr< T > & p_ten) {

		std::ifstream
		ifs(p_filename);

		if(! ifs.is_open())

			return false;

		bool
		ret = load(ifs, p_ten);

		ifs.close();

		return ret;
	}

	template< typename T = double >
	bool
	load(std::string const & p_filename, Tsr4< T > & p_ten) {

		std::ifstream
		ifs(p_filename);

		if(! ifs.is_open())

			return false;

		bool
		ret = load(ifs, p_ten);

		ifs.close();

		return ret;
	}

	template< typename T = double >
	T pop_back(Vec< T > & p_vec) {

		T v = p_vec[p_vec.size() - 1];

		p_vec.pop_back();

		return v;
	}

	template< typename T = double >
	T pop_front(Vec< T > & p_vec) {

		T v = p_vec[0];

		for(std::size_t i = 1; i < p_vec.size(); ++ i) {

			p_vec[i - 1] = p_vec[i];
		}

		p_vec.pop_back();

		return v;
	}

	template< typename T = double >
	Vec< T > & push_back(Vec< T > & p_vec, T const & p_val) {

		p_vec.push_back(p_val);

		return p_vec;
	}

	template< typename T = double >
	Vec< T > & push_front(Vec< T > & p_vec, T const & p_val) {
		if(0 < p_vec.size()) {
			p_vec.push_back(p_vec[p_vec.size() - 1]);
			for(std::size_t i = p_vec.size() - 1; 0 < i; -- i) {

				p_vec[i] = p_vec[i - 1];
			}
			p_vec[0] = p_val;
		} else {
			p_vec.push_back(p_val);
		}
		return p_vec;
	}

	template< typename T = double >
	Vec< T > combine(Vec< T > const & p_vec0, Vec< T > const & p_vec1) {

		Vec< T >
		v(p_vec0.size() + p_vec1.size());

		std::copy(p_vec0.cbegin(), p_vec0.cend(), v.begin());
		std::copy(p_vec1.cbegin(), p_vec1.cend(), v.begin() + p_vec0.size());

		return v;
	}

//	template< typename T = double >
//	Vec< T > remove(Vec< T > const & p_vec, std::size_t const & p_id) {
//		Vec< T >
//		v(p_vec.size() - 1);
//		std::copy(p_vec.cbegin(), p_vec.cbegin() + p_id, v.begin());
//		std::copy(p_vec.cbegin() + p_id + 1, p_vec.cend(), v.begin() + p_id);
//		return v;
//	}

    template< typename T = double >
    Vec< T >
    remove(Vec< T > const & p_vec, std::size_t const & p_id, std::size_t const & p_len = 1) {
        Vec< T >
        v(p_vec.size() - p_len);
        std::copy(p_vec.cbegin(), p_vec.cbegin() + p_id, v.begin());
        std::copy(p_vec.cbegin() + p_id + p_len, p_vec.cend(), v.begin() + p_id);
        return v;
    }

    template< typename T = double >
    Vec< T >
    insert_val(Vec< T > const & p_vec, std::size_t const & p_id, T const & p_val) {
        Vec< T >
        v(p_vec.size() + 1);
        std::copy(p_vec.cbegin(), p_vec.cbegin() + p_id, v.begin());
        v[p_id] = p_val;
        std::copy(p_vec.cbegin() + p_id, p_vec.cend(), v.begin() + p_id + 1);
        return v;
    }

    template< typename T = Vec< double > >
    T
    insert_vec(T const & p_vec1, std::size_t const & p_id, T const & p_vec2) {
        T
        v(p_vec1.size() + p_vec2.size());
        std::copy(p_vec1.cbegin(), p_vec1.cbegin() + p_id, v.begin());
        std::copy(p_vec2.cbegin(), p_vec2.cend(), v.begin() + p_id);
        std::copy(p_vec1.cbegin() + p_id, p_vec1.cend(), v.begin() + p_id + p_vec2.size());
        return v;
    }

    /*template< typename T = double>
    Vec<T> sgn(Vec<T> p_vec) {
        return std::transform(
            p_vec.begin(),
            p_vec.end(),
            p_vec.begin(),
            [](auto x) {
            return x < 0
                ? -1
                : 0 < x
                    ? 1
                    : 0;
            }
        );
    }
*/
    template <typename T>
    T sgn(T const & p_v) {
        return p_v < static_cast<T>(0)
            ? static_cast<T>(-1)
            : static_cast<T>(0) < p_v
                ? static_cast<T>(1)
                    : static_cast<T>(0);
    }

    template <typename T>
    Vec<T> sgnv(Vec<T> p_v) {
        std::transform(
            p_v.begin(),
            p_v.end(),
            p_v.begin(),
                [](auto x) {
                return x < 0
                    ? -1
                    : 0 < x
                        ? 1
                        : 0;
                }
        );
        return p_v;
    }


 //@even more abbr.
	typedef std::size_t UI;
	typedef double      D;
	typedef Vec< UI >   VU;
	typedef Vec< D >    VD;
	typedef Vec< VD >   MD;
	typedef Vec< MD >   TD;
//@
}
#endif // ALGEBRA_HPP
