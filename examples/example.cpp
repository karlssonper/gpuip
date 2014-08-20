#include <gpuip.h>

void print_timings(const char * func_name, double ms, std::string * err)
{
    if (ms != GPUIP_ERROR) {
        printf("%s took %.2lf ms.\n", func_name, ms);
    } else {
        printf("Error in %s: %s\n", func_name, err->c_str());
    }
}

void use_gpuip()
{
    std::string err;
    float * data;
    unsigned int width, height;
    ReadImage(&data, &width, &height); // definied somewhere else

    if (!gpuip::ImageProcessor::CanCreateGpuEnvironment(gpuip::GLSL)) {
        // ... deal with error - throw exception, return function etc
    }
    gpuip::ImageProcessor::Ptr ip = gpuip::ImageProcessor::Create(gpuip::GLSL);
    ip->SetDimensions(width, height);
    gpuip::Buffer::Ptr b0 = ip->CreateBuffer("b0", gpuip::FLOAT, 4);
    gpuip::Buffer::Ptr b1 = ip->CreateBuffer("b1", gpuip::FLOAT, 4);
    gpuip::Kernel::Ptr kernel = gpuipip->CreateKernel("modify_red");
    kernel->code = GetKernelCode(); // definied somewhere else
    kernel->inBuffers.push_back(gpuip::Kernel::BufferLink(b0, "img"));
    kernel->outBuffers.push_back(gpuip::Kernel::BufferLink(b1, "out_img"));
    kernel->paramsFloat.push_back(gpuip::Parameter<float>("alpha", 0.4));
    print_timings("Build", ip->Build(&err), &err);
    print_timings("Allocate", ip->Allocate(&err), &err);
    print_timings("Copy", ip->Copy(b0, gpuip::Buffer::COPY_TO_GPU, data, &err), &err);
    print_timings("Run", ip->Run(&err), &err);
    print_timings("Copy", ip->Copy(b1, gpuip::Buffer::COPY_FROM_GPU, data, &err), &err);
}
