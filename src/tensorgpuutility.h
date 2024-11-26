#ifndef TENSORGPUUTILITYH
#define TENSORGPUUTILITYH

#include <memory>

namespace TensorGPUUtility{
    std::shared_ptr<double> toGPU(double * data, size_t dataLen);

    void toCPU(double * data, double * d_data, size_t dataLen);

    std::shared_ptr<double> convert(std::shared_ptr<double> p, bool toGPU, size_t dataLen);

    std::shared_ptr<double> allocate(size_t dataLen);
}

#endif
