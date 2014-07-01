#include "../src/gpuip.h"
#include <cassert>

int main()
{
    gpuip::Base * b = gpuip::Factory::Create(gpuip::OpenCL,5, 5);
    
    return 0;
}
