clang++ -std=c++14 -ggdb -Wall -Wextra -pedantic -Wno-reorder-ctor src/tensor.cc src/tensorcontents.cc src/tensorcpufunctions.cc test1.cc -o test1
#clang++ -std=c++14 -Wall -Wextra -pedantic -ggdb -Wno-reorder-ctor -fPIC $(python3 -m pybind11 --includes) tensorpybind.h -o tensor$(python3-config --extension-suffix)
#nvcc -forward-unknown-to-host-compiler -Wall -Wextra -pedantic -std=c++14 -arch=sm_61 -DCUDA tensor.cc tensorcontents.cc tensorcpufunctions.cc tensorgpuutility.cc tensorgpufunctions.cu test1.cc -o test1
#nvcc -forward-unknown-to-host-compiler -std=c++14 -arch=sm_61 -DCUDA tensor.cc tensorcontents.cc tensorcpufunctions.cc tensorgpuutility.cc tensorgpufunctions.cu test1.cc -o test1
#g++ -std=c++14 -ggdb -Wall -Wextra -pedantic -fopenmp -DOMP tensor.cc tensorcontents.cc tensorcpufunctions.cc test1.cc -o test1
#g++ -std=c++14 -ggdb -fopenmp -DOMP tensor.cc tensorcontents.cc tensorcpufunctions.cc test1.cc -o test1
