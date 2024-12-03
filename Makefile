
tensor:
	clang++ -std=c++14 -O3 src/tensor.cc src/tensorcontents.cc src/tensorcpufunctions.cc mnist_demo.cc -o main

tensor-omp:
	g++ -std=c++14 -fopenmp -DOMP src/tensor.cc src/tensorcontents.cc src/tensorcpufunctions.cc mnist_demo.cc -o main

tensor-cuda:
	nvcc -std=c++14 -arch=sm_61 -DCUDA src/tensor.cc src/tensorcontents.cc src/tensorcpufunctions.cc mnist_demo.cc -o main

tensor-omp-cuda:
	nvcc -std=c++14 -arch=sm_61 -DCUDA -XCompiler -fopenmp -DOMP src/tensor.cc src/tensorcontents.cc src/tensorcpufunctions.cc mnist_demo.cc -o main

