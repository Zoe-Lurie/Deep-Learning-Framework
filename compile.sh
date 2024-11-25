clang++ -std=c++11 -ggdb -Wall -Wextra -pedantic -Wno-reorder-ctor tensor.cc tensorcontents.cc tensorcpufunctions.cc test1.cc -o test1
#clang++ -std=c++17 -Wall -Wextra -pedantic -ggdb -fPIC $(python3 -m pybind11 --includes) tensor.cc tensorutility.cc tensoreval.cc tensorpybind.h -o tensor$(python3-config --extension-suffix)

