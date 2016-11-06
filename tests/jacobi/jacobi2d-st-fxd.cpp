#include <cstdlib>

#include "include/jacobi-stencil.hpp"

class kernel_st;
class kernel_copy;

class out_dummy;
class in_dummy;

template<>
inline float& f<float,out_dummy>(int a,int b, cl::sycl::accessor<float, 2, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer> acc) {return acc[a][b];}
template<>
inline float a_f<float,in_dummy>(int a,int b, cl::sycl::accessor<float, 2, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer>  acc) {return acc[a][b];}

// static declaration to use pointers
cl::sycl::buffer<float,2> ioBuffer = cl::sycl::buffer<float,2>(cl::sycl::range<2> {M, N});
cl::sycl::buffer<float,2> ioABuffer = cl::sycl::buffer<float,2>(cl::sycl::range<2> {M, N});

int main(int argc, char **argv) {
  //  read_args(argc, argv);
  struct counters timer;
  start_measure(timer);

  // declarations 
#if DEBUG_STENCIL
  float *a_test = (float *) malloc(sizeof(float)*M*N);
  float *b_test = (float *) malloc(sizeof(float)*M*N);
#endif

  // initialization
  for (size_t i = 0; i < M; ++i){
    for (size_t j = 0; j < N; ++j){
      float value = ((float) i*(j+2) + 10) / N;
      cl::sycl::id<2> id = {i, j};
      ioBuffer.get_access<cl::sycl::access::mode::write, cl::sycl::access::target::host_buffer>()[id] = value;
      ioABuffer.get_access<cl::sycl::access::mode::write, cl::sycl::access::target::host_buffer>()[id] = value;
#if DEBUG_STENCIL
      a_test[i*N+j] = value;
      b_test[i*N+j] = value;
#endif
    }
  }

  // our work
  coef_fxd2D<0,0> c_id {1.0f};
  coef_fxd2D<0, 0> c1 {MULT_COEF};  
  coef_fxd2D<1, 0> c2 {MULT_COEF};
  coef_fxd2D<0, 1> c3 {MULT_COEF};
  coef_fxd2D<-1, 0> c4 {MULT_COEF};
  coef_fxd2D<0, -1> c5 {MULT_COEF};

  auto st = c1+c2+c3+c4+c5;
  input_fxd2D<float, &ioABuffer, in_dummy> work_in;
  output_fxd2D<float, &ioBuffer, out_dummy> work_out;
  auto op_work = work_out << st << work_in;

  auto st_id = c_id.toStencil();
  input_fxd2D<float, &ioBuffer, in_dummy> copy_in;
  output_fxd2D<float, &ioABuffer, out_dummy> copy_out;
  auto op_copy = copy_out << st_id << copy_in;

  end_init(timer);
  struct op_time time_op;
  begin_op(time_op);

  // compute result with "gpu"
  {   
    cl::sycl::gpu_selector selector;
    cl::sycl::queue myQueue(selector); 
    for (unsigned int i = 0; i < NB_ITER; ++i){      
      //op_work.doComputation(myQueue);
      op_work.template doLocalComputation<kernel_st>(myQueue);
      op_copy.template doComputation<kernel_copy>(myQueue);
    }
  }

  end_op(time_op, timer.stencil_time);
  // loading time is not watched
  end_measure(timer);

#if DEBUG_STENCIL
  // get the gpu result
  auto C = ioABuffer.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>();
  ute_and_are(a_test,b_test,C);
  free(a_test);
  free(b_test);
#endif

  return 0;
}
