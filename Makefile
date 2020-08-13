CC = g++
NVCC = nvcc
CFLAGS = -fpic -fopenmp
LDFLAGS = -shared
CUDA_FLAGS = -arch=sm_70 -Xcompiler '-fPIC'
CUDA_LDFLAGS = -L/usr/local/cuda/lib64 -lcudart

all: libinterp.so libkernel.so

libinterp.so: interp.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(CUDA_LDFLAGS)

libkernel.so: kernel.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

interp.o: interp.cu
	$(NVCC) -c -o $@ $^ $(CUDA_FLAGS)

kernel.o: kernel.c
	$(CC) -c -o $@ $^ $(CFLAGS) 

clean:
	rm -f $(TARGET) *.o
