CC = g++
NVCC = nvcc
CFLAGS = -fpic -fopenmp -m64 -march=native -mtune=native
LDFLAGS = -shared
MKL_CFLAGS = -DMKL_ILP64 -I${MKLROOT}/include 
MKL_LDFLAGS = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
	${MKLROOT}/lib/intel64/libmkl_intel_thread.a \
	${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group \
	-liomp5 -lpthread -lm -ldl
CUDA_FLAGS = -arch=sm_70 -Xcompiler '-fPIC'
CUDA_LDFLAGS = -L/usr/local/cuda/lib64 -lcudart

all: libinterp.so libkernel.so liblinsys.so

libinterp.so: interp.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

libkernel.so: kernel.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

liblinsys.so: linsys.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(MKL_LDFLAGS)

# interp.o: interp.cu
# 	$(NVCC) -c -o $@ $^ $(CUDA_FLAGS)
interp.o: interp.cpp
	$(CC) -c -o $@ $^ $(CFLAGS) -Ofast

kernel.o: kernel.cpp
	$(CC) -c -o $@ $^ $(CFLAGS) -Ofast

linsys.o: linear_system.cpp
	$(CC) -c -o $@ $^ $(CFLAGS) $(MKL_CFLAGS) -O3

clean:
	rm -f *.so *.o
