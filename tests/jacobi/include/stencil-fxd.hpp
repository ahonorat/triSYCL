#ifndef STENCIL_INRIA_FXD
#define STENCIL_INRIA_FXD

#include "stencil-common.hpp"

template <typename T, cl::sycl::buffer<T,2> *_B, class out_fdl>
class output_fxd2D {};

template <typename T, class in_fdl>
inline T a_f (int a,int b, cl::sycl::accessor<T, 2, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer> buf);
template <typename T, class out_fdl>
inline T& f (int a,int b, cl::sycl::accessor<T, 2, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer> buf);



// fixed coeff

class auth_in_st_fxd {
protected:
  auth_in_st_fxd () {}
};

class auth_in_op_fxd {
protected:
  auth_in_op_fxd () {}
};

template <class c_or0_s2D, class c_or1_s2D, typename T> 
class stencil_fxd2D : private auth_in_st_fxd, private auth_in_op_fxd {
public:
  static_assert(std::is_base_of<auth_in_st_fxd, c_or0_s2D>::value,"A stencil must be built from a coef or a stencil.");
  static_assert(std::is_base_of<auth_in_st_fxd, c_or1_s2D>::value,"A stencil must be built from a coef or a stencil.");
  static const int min_ind0 = MIN((c_or0_s2D::min_ind0), (c_or1_s2D::min_ind0));
  static const int max_ind0 = MAX((c_or0_s2D::max_ind0), (c_or1_s2D::max_ind0));
  static const int min_ind1 = MIN((c_or0_s2D::min_ind1), (c_or1_s2D::min_ind1));
  static const int max_ind1 = MAX((c_or0_s2D::max_ind1), (c_or1_s2D::max_ind1));

  stencil_fxd2D(c_or0_s2D st0, c_or1_s2D st1) : st_left(st0), st_right(st1) {}

  template<class in_fdl>
  inline T eval(cl::sycl::accessor<T, 2, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer> a, int k, int l) const {
    return st_left.template eval<T,in_fdl>(a, a_f, k, l) + st_right.template eval<T,in_fdl>(a, k, l);
  }
  template<int ldc>
  inline T eval_local(cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> a, int k_local, int l_local) const {
    return st_left.template eval_local<ldc>(a, k_local, l_local) + st_right.template eval_local<ldc>(a, k_local, l_local);
  }
private:
  const c_or0_s2D st_left;
  const c_or1_s2D st_right;
};

template <class c_or0_s2D, typename T = float>
class stencil_fxd2D_bis : private auth_in_st_fxd, private auth_in_op_fxd{
public:
  static_assert(std::is_base_of<auth_in_st_fxd, c_or0_s2D>::value,"A stencil must be built from a coef or a stencil.");
  static const int min_ind0 = c_or0_s2D::min_ind0;
  static const int max_ind0 = c_or0_s2D::max_ind0;
  static const int min_ind1 = c_or0_s2D::min_ind1;
  static const int max_ind1 = c_or0_s2D::max_ind1;

  stencil_fxd2D_bis(c_or0_s2D st0) : st(st0) {}

  template<class in_fdl>
  inline T eval(cl::sycl::accessor<T, 2, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer> a, int k, int l) const {
    return st.template eval<in_fdl>(a, k, l);
  }
  template <int ldc>
  inline T eval_local(cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> a, int k_local, int l_local) const {
    return st.template eval_local<ldc>(a, k_local, l_local);
  }
private:
  const c_or0_s2D st;
};

template <int i, int j, typename T = float>
class coef_fxd2D : private auth_in_st_fxd {
public:
  static const int min_ind0 = i;
  static const int max_ind0 = i;
  static const int min_ind1 = j;
  static const int max_ind1 = j;

  coef_fxd2D(T a) : coef(a) {}

  inline stencil_fxd2D_bis<coef_fxd2D<i, j>> toStencil() {
    return stencil_fxd2D_bis<coef_fxd2D<i, j>> {*this};
  }

  template<class in_fdl>
  inline T eval(cl::sycl::accessor<T, 2, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer> a, int k, int l) const {
    return coef * a_f<T,in_fdl>(k+i,l+j,a); // template operator ? it would be cool
  }
  template<int ldc>
  inline T eval_local(cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> a, int k_local, int l_local) const {
    return coef * a[(k_local+i)*ldc + l_local+j];
  }
private:
  const T coef;
};

template <int i, int j, int k, int l, typename T>
inline stencil_fxd2D<coef_fxd2D<i, j, T>, coef_fxd2D<k, l, T>, T> operator+ (coef_fxd2D<i, j, T> coef0, coef_fxd2D<k, l, T> coef1) {
  return stencil_fxd2D<coef_fxd2D<i, j, T>, coef_fxd2D<k, l, T>, T> {coef0, coef1};
}

template <int k, int l, class C1, class C2, typename T>
inline stencil_fxd2D<stencil_fxd2D<C1, C2, T>, coef_fxd2D<k, l, T>, T> operator+ (stencil_fxd2D<C1, C2, T> st0, coef_fxd2D<k, l, T> coef1) {
  return stencil_fxd2D<stencil_fxd2D<C1, C2, T>, coef_fxd2D<k, l, T>, T> {st0, coef1};
}

template <int k, int l, class C1, class C2, typename T>
inline stencil_fxd2D<coef_fxd2D<k, l, T>, stencil_fxd2D<C1, C2, T>, T> operator+ (coef_fxd2D<k, l, T> coef0, stencil_fxd2D<C1, C2, T> st1) {
  return stencil_fxd2D<coef_fxd2D<k, l, T>, stencil_fxd2D<C1, C2, T>, T> {coef0, st1};
}

template <class C1, class C2, class C3, class C4, typename T>
inline stencil_fxd2D<stencil_fxd2D<C1, C2, T>, stencil_fxd2D<C3, C4, T>, T> operator+ (stencil_fxd2D<C1, C2, T> st0, stencil_fxd2D<C3, C4, T> st1) {
  return stencil_fxd2D<stencil_fxd2D<C1, C2, T>, stencil_fxd2D<C3, C4, T>, T> {st0, st1};
}

template <typename T, cl::sycl::buffer<T,2> *_aB, class in_fdl>
class input_fxd2D {};

template <typename T, class st, cl::sycl::buffer<T,2> *_aB, class in_fdl>
class stencil_input_fxd2D {
public:
  const st stencil;
  stencil_input_fxd2D(st sten) : stencil(sten) {}
};


template <typename T, cl::sycl::buffer<T,2> *B, class out_fdl, class st, cl::sycl::buffer<T,2> *aB, class in_fdl>
class operation_fxd2D {
public:
  static_assert(std::is_base_of<auth_in_op_fxd, st>::value, "An operation must be built with a stencil.");
  static const int nb_tab_read = 1;
  static const int nb_tab_write = 1;
  static const int nb_op = 1;

  static const int d0 = st::max_ind0-st::min_ind0;
  static const int d1 = st::max_ind1-st::min_ind1;

  // offsets ands paddings for global memory
  static const int of0 = MAX(-st::min_ind0, 0); //offset left
  static const int of1 = MAX(-st::min_ind1, 0); //offset top
  static const int pad0 = MAX(0, st::max_ind0); //padding right
  static const int pad1 = MAX(0, st::max_ind1); //padding bottom

  // offset for local tile
  // ajusted in 0
  static const int dev0 = st::min_ind0;
  static const int dev1 = st::min_ind1;

  static const local_info2D<T, nb_tab_read, d0, d1> li2D;
  static const int local_dim0 = li2D.nbi_wg0 + d0;
  static const int local_dim1 = li2D.nbi_wg1 + d1;

  cl::sycl::range<2> d = {d0, d1};
  cl::sycl::id<2> offset = {0, 0};
  cl::sycl::range<2> range;
  cl::sycl::nd_range<2> nd_range = {cl::sycl::range<2> {}, cl::sycl::range<2> {}, cl::sycl::id<2> {}};

  int global_max0;
  int global_max1;

  const st stencil;

  operation_fxd2D(st sten) : stencil(sten) {    
    cl::sycl::range<2> rg1 = aB->get_range();
    cl::sycl::range<2> rg2 = B->get_range();
    assert(rg1 == rg2); //seems a bad idea, indeed ! we will not do that ...
    range = rg1-d;
    int r0 = range.get(0);
    int r1 = range.get(1);
    global_max0 = r0;
    global_max1 = r1;
    if (r0 % li2D.nbi_wg0 != 0) {
      r0 = ((global_max0 / li2D.nbi_wg0) + 1) * li2D.nbi_wg0;
    }
    if (r1 % li2D.nbi_wg1 != 0) {
      r1 = ((global_max1 / li2D.nbi_wg1) + 1) * li2D.nbi_wg1;
    }
    nd_range = {cl::sycl::range<2> {(unsigned int) r0, (unsigned int) r1}, cl::sycl::range<2> {li2D.nbi_wg0, li2D.nbi_wg1}, offset};
    //    nd_range = {range, cl::sycl::range<2> {li2D.nbi_wg0, li2D.nbi_wg1}, offset};
  }

  static inline void eval(st _stencil, cl::sycl::id<2> id, cl::sycl::accessor<T, 2, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer> out, cl::sycl::accessor<T, 2, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer> in) {
    int i = id.get(0) + of0;
    int j = id.get(1) + of1;
    f<T,out_fdl>(i, j, out) = _stencil.template eval<in_fdl>(in, i, j);
  }

  static inline void eval_local(st _stencil, cl::sycl::nd_item<2> it, cl::sycl::accessor<T, 2, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer> out, cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> local_tab, int glob_max0, int glob_max1) {
    int i = it.get_global().get(0); 
    int j = it.get_global().get(1);
    if (i >= glob_max0 || j >= glob_max1)
      return;
    i += of0;
    j += of1;
    int i_local = it.get_local().get(0) - st::min_ind0;
    int j_local = it.get_local().get(1) - st::min_ind1;
    f<T,out_fdl>(i, j, out) = _stencil.template eval_local<local_dim1>(local_tab, i_local, j_local);
  }

  static inline void store_local(cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> local_tab, cl::sycl::accessor<T, 2, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer> in, cl::sycl::nd_item<2> it, cl::sycl::group<2> gr, int glob_max0, int glob_max1) {
    cl::sycl::range<2> l_range = it.get_local_range();
    cl::sycl::id<2> g_ind = gr.get(); //it.get_group_id(); error because ambiguous / operator redefinition 
    cl::sycl::id<2> l_ind = it.get_local();

    int l_range0 = l_range.get(0);
    int l_range1 = l_range.get(1);
    int l_ind0 = l_ind.get(0);
    int l_ind1 = l_ind.get(1);
    int gr_ind0 = g_ind.get(0);
    int gr_ind1 = g_ind.get(1);
      
    int block_dim0 = local_dim0 / l_range0;
    int block_dim1 = local_dim1 / l_range1;
    int total_block_dim0 = block_dim0;
    int total_block_dim1 = block_dim1;
    if (l_ind0 == l_range0 - 1)
      total_block_dim0 += local_dim0 % l_range0;
    if (l_ind1 == l_range1 - 1)
      total_block_dim1 += local_dim1 % l_range1;

    int local_ind0 = l_ind0 * block_dim0;
    int local_ind1 = l_ind1 * block_dim1;
    int global_ind0 = gr_ind0 * l_range0 + local_ind0 + of0 + st::min_ind0;
    int global_ind1 = gr_ind1 * l_range1 + local_ind1 + of1 + st::min_ind1;

    for (int i = 0; i < total_block_dim0; ++i){
      int j;
      for (j = 0; j < total_block_dim1; ++j){
	if (global_ind0 < glob_max0 && global_ind1 < glob_max1)
	  local_tab[local_ind0 * local_dim1 + local_ind1] = a_f<T,in_fdl>(global_ind0, global_ind1, in);
	local_ind1++;
	global_ind1++;
      }
      local_ind0++;
      global_ind0++;
      local_ind1 -= total_block_dim1;
      global_ind1 -= total_block_dim1;
    }
  }

  template<class KernelName>
  inline void doComputation(cl::sycl::queue queue){
    queue.submit([&](cl::sycl::handler &cgh) {
	cl::sycl::accessor<T, 2, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer> _B{*B, cgh};
	cl::sycl::accessor<T, 2, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer> _aB{*aB, cgh};
	auto _stencil = stencil;
	cgh.parallel_for<KernelName>(range, [=](cl::sycl::id<2> id){
	    eval(_stencil, id, _B, _aB);
	  });
      });
  }

  template<class KernelName>
  inline void doLocalComputation(cl::sycl::queue queue){
    queue.submit([&](cl::sycl::handler &cgh) {
	cl::sycl::accessor<T, 2, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer> _B{*B, cgh};
	cl::sycl::accessor<T, 2, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer> _aB{*aB, cgh};
	cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> local{cl::sycl::range<1>(local_dim0 * local_dim1), cgh};
        auto _global_max0 = global_max0;
	auto _global_max1 = global_max1;
	auto _stencil = stencil;
	auto kernel = [=](cl::sycl::group<2> group){
	    cl::sycl::parallel_for_work_item(group, [=](cl::sycl::nd_item<2> it){
		//local copy
		/* group shoudn't be needed, neither global max*/
		/* static function needed for st use a priori, but static not compatible
		   with dynamic filed as global_max */
		store_local(local, _aB, it, group, _global_max0+d0, _global_max1+d1); 
	      });
	    //synchro
	    cl::sycl::parallel_for_work_item(group, [=](cl::sycl::nd_item<2> it){
		//computing
		/*operation_fxd2D<T, B, f, st, aB, bB, a_f, b_f>::*/
		eval_local(_stencil, it, _B, local, _global_max0, _global_max1);
	      });
	  };
	cgh.parallel_for_work_group<KernelName>(nd_range, kernel);
      });
  }


};


template <typename T, cl::sycl::buffer<T,2> *_B, class out_fdl, class st>
class output_stencil_fxd2D {
public:
  const st stencil;
  output_stencil_fxd2D(st sten) : stencil(sten) {}
};



template <typename T, cl::sycl::buffer<T,2> *B, class out_fdl, class C1, class C2>
inline output_stencil_fxd2D<T, B, out_fdl, stencil_fxd2D<C1, C2, T>> operator<< (output_fxd2D<T, B, out_fdl> out, stencil_fxd2D<C1, C2, T> in) {
  return output_stencil_fxd2D<T, B, out_fdl, stencil_fxd2D<C1, C2, T>> {in};
}

template <typename T, cl::sycl::buffer<T,2> *B, class out_fdl, class C1>
inline output_stencil_fxd2D<T, B, out_fdl, stencil_fxd2D_bis<C1, T>> operator<< (output_fxd2D<T, B, out_fdl> out, stencil_fxd2D_bis<C1, T> in) {
  return output_stencil_fxd2D<T, B, out_fdl, stencil_fxd2D_bis<C1, T>> {in};
}



template <typename T, class C1, class C2, cl::sycl::buffer<T,2> *aB, class in_fdl>
inline stencil_input_fxd2D<T, stencil_fxd2D<C1, C2, T>, aB, in_fdl> operator<< (stencil_fxd2D<C1, C2, T> out, input_fxd2D<T, aB, in_fdl> in) {
  return stencil_input_fxd2D<T, stencil_fxd2D<C1, C2, T>, aB, in_fdl> {out};
}

template <typename T, class C1, cl::sycl::buffer<T,2> *aB, class in_fdl>
inline stencil_input_fxd2D<T, stencil_fxd2D_bis<C1, T>, aB, in_fdl> operator<< (stencil_fxd2D_bis<C1, T> out, input_fxd2D<T, aB, in_fdl> in) {
  return stencil_input_fxd2D<T, stencil_fxd2D_bis<C1, T>, aB, in_fdl> {out};
}



template <typename T, cl::sycl::buffer<T,2> *B, class out_fdl, class st, cl::sycl::buffer<T,2> *aB, class in_fdl>
inline operation_fxd2D<T, B, out_fdl, st, aB, in_fdl> operator<< (output_stencil_fxd2D<T, B, out_fdl, st> out, input_fxd2D<T, aB, in_fdl> in) {
  return operation_fxd2D<T, B, out_fdl, st, aB, in_fdl> {out.stencil};
}

template <typename T, cl::sycl::buffer<T,2> *B, class out_fdl, class st, cl::sycl::buffer<T,2> *aB, class in_fdl>
inline operation_fxd2D<T, B, out_fdl, st, aB, in_fdl> operator<< (output_fxd2D<T, B, out_fdl> out, stencil_input_fxd2D<T, st, aB, in_fdl> in) {
  return operation_fxd2D<T, B, out_fdl, st, aB, in_fdl> {in.stencil};
}

#endif

