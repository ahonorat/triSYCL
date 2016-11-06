#include <cstdlib>
#include <CL/sycl.hpp>

#include <iostream>

using namespace cl;

size_t M = 6, N = 10;
size_t tile0 = 4, tile1 = 4;


sycl::id<2> id_add(sycl::id<2> a, sycl::id<2> b) {
  return sycl::id<2>(a.get(0) + b.get(0), a.get(1) + b.get(1));
}

sycl::id<2> id_sub(sycl::id<2> a, sycl::id<2> b) {
  return sycl::id<2>(a.get(0) - b.get(0), a.get(1) - b.get(1));
}

void print_buffer2D(sycl::accessor<float, 2, sycl::access::mode::read, sycl::access::target::host_buffer> bufferAccessor) {
  for (size_t i = 0; i < M; ++i){
    for(size_t j = 0; j < N; ++j){
      std::cout << bufferAccessor[i][j] << " " ;
    }
    std::cout << std::endl;
  }
}

int main(int argc, char **argv) {
  // declarations
  sycl::buffer<float,2> ioABuffer = sycl::buffer<float,2>(sycl::range<2> {M, N});
  sycl::buffer<float,2> ioBBuffer = sycl::buffer<float,2>(sycl::range<2> {M, N}); 
  sycl::buffer<float,2> ioCBuffer = sycl::buffer<float,2>(sycl::range<2> {M, N}); 

  
  // initialization
  for (size_t i = 0; i < M; ++i){
    for (size_t j = 0; j < N; ++j){
      float value = ((float) i*(j+2) + 10) / N;
      sycl::id<2> id = {i, j};
      ioABuffer.get_access<sycl::access::mode::write, sycl::access::target::host_buffer>()[id] = value;
      ioBBuffer.get_access<sycl::access::mode::write, sycl::access::target::host_buffer>()[id] = value;
      ioCBuffer.get_access<sycl::access::mode::write, sycl::access::target::host_buffer>()[id] = value;
    }
  }
  
  // compute result with gpu
  {  
    sycl::queue myQueue; 

    myQueue.submit([&](sycl::handler &cgh) {
	sycl::accessor<float, 2, sycl::access::mode::read, sycl::access::target::global_buffer>  a(ioABuffer, cgh);
	sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> b(ioBBuffer, cgh);
	sycl::accessor<float, 2, sycl::access::mode::write, sycl::access::target::global_buffer> c(ioCBuffer, cgh);
	sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> local(sycl::range<1>((tile0+2)*(tile1+2)), cgh);
	auto cl0 = tile0;
	auto cl1 = tile1;
	cgh.parallel_for_work_group<class KernelCompute>(sycl::nd_range<2> {
	    sycl::range<2> {M-2, N-2}, 
	      sycl::range<2> {tile0, tile1},
		sycl::id<2> {0, 0}},
	  [=](sycl::group<2> group){
	    sycl::parallel_for_work_item(group, [=](sycl::nd_item<2> it){
		sycl::range<2> l_range = it.get_local_range();
		sycl::id<2> g_ind = it.get_global();
		sycl::id<2> l_ind = it.get_local();
		sycl::id<2> offset = sycl::id<2> {1, 1};
		sycl::id<2> id1(sycl::range<2> {0,1});
		sycl::id<2> id2(sycl::range<2> {1,0});
		sycl::id<2> id1_s(sycl::range<2> {0,l_range.get(1)});
		sycl::id<2> id2_s(sycl::range<2> {l_range.get(0),0});
		local[(id_add(l_ind,offset)).get(0)*(cl1+2) + (id_add(l_ind,offset)).get(1)] = a[id_add(g_ind,offset)];
		if (l_ind.get(0) == 0) {
		  local[(id_add(l_ind,offset)).get(1)] = a[id_sub(id_add(g_ind,offset),id2)];
		  local[(id_add(id2_s,offset)).get(0)*(cl1+2) + (id_add(l_ind,offset)).get(1)] = a[id_add(id_add(g_ind,id2_s),offset)];
		}
		if (l_ind.get(1) == 0) {
		  local[(id_add(l_ind,offset)).get(0)*(cl1+2)] = a[id_sub(id_add(g_ind,offset),id1)];
		  local[(id_add(l_ind,offset)).get(0)*(cl1+2) + (id_add(id1_s,offset)).get(1)] = a[id_add(id_add(g_ind,id1_s),offset)];
		}
	      });

	    sycl::parallel_for_work_item(group, [=](sycl::nd_item<2> it){
		sycl::id<2> g_ind = it.get_global();
		sycl::id<2> l_ind = it.get_local();
		sycl::id<2> offset = sycl::id<2> {1, 1};
		sycl::id<2> id1(sycl::range<2> {0,1});
		sycl::id<2> id2(sycl::range<2> {1,0});
		// b[id_add(g_ind,offset)] = local[(id_add(l_ind,offset)).get(0)*(cl1+2) + (id_add(l_ind,offset)).get(1)]; // print good values
		// c[id_add(g_ind,offset)] = local[0] + local[1]; // do kind of nonsense (see printing)
		// // and if b and c are both uncommented, i'll get an error at runtime, kernel cannot be built
		// // I also have this message: "WARNING: Linking two modules of different data layouts!" why ?
		// ==> ok corrected with -O2
		b[id_add(g_ind,offset)] = local[(id_add(l_ind,offset)).get(0)*(cl1+2) + (id_add(l_ind,offset)).get(1)];
		// b has same values as a
		c[id_add(g_ind,offset)] = local[(id_add(l_ind,offset)).get(0)*(cl1+2) + (id_add(l_ind,offset)).get(1)]
		  + local[(id_add(l_ind,offset)).get(0)*(cl1+2) + (id_add(l_ind,offset)).get(1)];
		// all values should be doubled in c, but only the first tile is indeed doubled !
	      });       	
	  });
	  
      });
  }
  
  print_buffer2D(ioABuffer.get_access<sycl::access::mode::read, sycl::access::target::host_buffer>());
  std::cout << "=======================================================================" << std::endl;
  print_buffer2D(ioBBuffer.get_access<sycl::access::mode::read, sycl::access::target::host_buffer>());
  std::cout << "=======================================================================" << std::endl;
  print_buffer2D(ioCBuffer.get_access<sycl::access::mode::read, sycl::access::target::host_buffer>());
  
  return 0;
}
