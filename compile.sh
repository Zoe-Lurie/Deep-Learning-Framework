clang++ -std=c++17 -Wall -Wextra -pedantic -ggdb tensor.cc tensorfunction.cc tensordata.cc tensorcontents.cc test1.cc -o test1
#clang++ -std=c++17 -Wall -Wextra -pedantic -ggdb -fPIC $(python3 -m pybind11 --includes) tensor.cc tensorutility.cc tensoreval.cc tensorpybind.h -o tensor$(python3-config --extension-suffix)

