#include <gpuip.h>
#include <cassert>
#include <stdlib.h>
#include <math.h>
//----------------------------------------------------------------------------//
const char * opencl_codeA = ""
" __kernel void                                                              \n"
"my_kernelA(__global const float * A,                                        \n"
"           __global float * B,                                              \n"
"           __global float * C,                                              \n"
"           const int incA,                                                  \n"
"           const float incB,                                                \n"
"           const int width,                                                 \n"
"           const int height)                                                \n"
"{                                                                           \n"
"    const int x = get_global_id(0);                                         \n"
"    const int y = get_global_id(1);                                         \n"
"                                                                            \n"
"    // array index                                                          \n"
"    const int idx = x + width * y;                                          \n"
"                                                                            \n"
"    // inside image bounds check                                            \n"
"    if (x >= width || y >= height) {                                        \n" 
"        return;                                                             \n"
"    }                                                                       \n"
"                                                                            \n"
"    // kernel code                                                          \n"
"    B[idx] = A[idx] + incA *0.1;                                            \n"
"    C[idx] = A[idx] + incB;                                                 \n"
"}";
const char * opencl_codeB = ""
"__kernel void                                                               \n"
"my_kernelB(__global const float * B,                                        \n"
"           __global const float * C,                                        \n"
"           __global float * A,                                              \n"
"           const int width,                                                 \n"
"           const int height)                                                \n"
"{                                                                           \n"
"    const int x = get_global_id(0);                                         \n"
"    const int y = get_global_id(1);                                         \n"
"                                                                            \n"
"    // array index                                                          \n"
"    const int idx = x + width * y;                                          \n"
"                                                                            \n"
"    // inside image bounds check                                            \n"
"    if (x >= width || y >= height) {                                        \n"
"        return;                                                             \n"
"    }                                                                       \n"
"                                                                            \n"
"    // kernel code                                                          \n"
"    A[idx] =  B[idx] + C[idx];                                              \n"
"}";
//----------------------------------------------------------------------------//
const char * opencl_boilerplateA = ""
"__kernel void\n"
"my_kernelA(__global const float * A,\n"
"           __global float * B,\n"
"           __global float * C,\n"
"           const int incA,\n"
"           const float incB,\n"
"           const int width,\n"
"           const int height)\n"
"{\n"
"    const int x = get_global_id(0);\n"
"    const int y = get_global_id(1);\n"
"\n"
"    // array index\n"
"    const int idx = x + width * y;\n"
"\n"
"    // inside image bounds check\n"
"    if (x >= width || y >= height) {\n" 
"        return;\n"
"    }\n"
"\n"
"    // kernel code\n"
"    B[idx] = 0;\n"
"    C[idx] = 0;\n"
"}";
const char * opencl_boilerplateB = ""
"__kernel void\n"
"my_kernelB(__global const float * B,\n"
"           __global const float * C,\n"
"           __global float * A,\n"
"           const int width,\n"
"           const int height)\n"
"{\n"
"    const int x = get_global_id(0);\n"
"    const int y = get_global_id(1);\n"
"\n"
"    // array index\n"
"    const int idx = x + width * y;\n"
"\n"
"    // inside image bounds check\n"
"    if (x >= width || y >= height) {\n"
"        return;\n"
"    }\n"
"\n"
"    // kernel code\n"
"    A[idx] = 0;\n"
"}";
//----------------------------------------------------------------------------//
const char * cuda_codeA = ""
"__global__ void                                                             \n"
"my_kernelA(const float * A,                                                 \n"
"           float * B,                                                       \n"
"           float * C,                                                       \n"
"           const int incA,                                                  \n"
"           const float incB,                                                \n"
"           const int width,                                                 \n"
"           const int height)                                                \n"
"{                                                                           \n"
"    const int x = blockIdx.x * blockDim.x + threadIdx.x;                    \n"
"    const int y = blockIdx.y * blockDim.y + threadIdx.y;                    \n"
"                                                                            \n"
"    // array index                                                          \n"
"    const int idx = x + width * y;                                          \n"
"                                                                            \n"
"    // inside image bounds check                                            \n"
"    if (x >= width || y >= height) {                                        \n"
"        return;                                                             \n"
"    }                                                                       \n"
"                                                                            \n"
"    // kernel code                                                          \n"
"    B[idx] = A[idx] + incA * 0.1;                                           \n"
"    C[idx] = A[idx] + incB;                                                 \n"
"}";
const char * cuda_codeB = ""
"__global__ void                                                             \n"
"my_kernelB(float * B,                                                       \n"
"           float * C,                                                       \n"
"           float * A,                                                       \n"
"           const int width,                                                 \n"
"           const int height)                                                \n"
"{                                                                           \n"
"    const int x = blockIdx.x * blockDim.x + threadIdx.x;                    \n"
"    const int y = blockIdx.y * blockDim.y + threadIdx.y;                    \n"
"                                                                            \n"
"    // array index                                                          \n"
"    const int idx = x + width * y;                                          \n"
"                                                                            \n"
"    // inside image bounds check                                            \n"
"    if (x >= width || y >= height) {                                        \n"
"        return;                                                             \n"
"    }                                                                       \n"
"                                                                            \n"
"    // kernel code                                                          \n"
"    A[idx] = B[idx] + C[idx];                                               \n"
"}";
//----------------------------------------------------------------------------//
const char * cuda_boilerplateA = ""
"__global__ void\n"
"my_kernelA(const float * A,\n"
"           float * B,\n"
"           float * C,\n"
"           const int incA,\n"
"           const float incB,\n"
"           const int width,\n"
"           const int height)\n"        
"{\n"
"    const int x = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    const int y = blockIdx.y * blockDim.y + threadIdx.y;\n"
"\n"
"    // array index\n"
"    const int idx = x + width * y;\n"
"\n"
"    // inside image bounds check\n"
"    if (x >= width || y >= height) {\n"
"        return;\n"
"    }\n"
"\n"
"    // kernel code\n"
"    B[idx] = 0;\n"
"    C[idx] = 0;\n"
"}";
const char * cuda_boilerplateB = ""
"__global__ void\n"
"my_kernelB(const float * B,\n"
"           const float * C,\n"
"           float * A,\n"
"           const int width,\n"
"           const int height)\n"         
"{\n"
"    const int x = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    const int y = blockIdx.y * blockDim.y + threadIdx.y;\n"
"\n"
"    // array index\n"
"    const int idx = x + width * y;\n"
"\n"
"    // inside image bounds check\n"
"    if (x >= width || y >= height) {\n"
"        return;\n"
"    }\n"
"\n"
"    // kernel code\n"
"    A[idx] = 0;\n"
"}";
//----------------------------------------------------------------------------//
const char * glsl_codeA = ""
"#version 120\n"
"uniform sampler2D A;                                                       \n"
"uniform int incA;                                                          \n"
"uniform float incB;                                                        \n"
"varying vec2 x; // texture coordinates                                     \n"
"uniform float dx; // delta                                                 \n"
"\n"        
"void main()                                                                \n"
"{                                                                          \n"
"    gl_FragData[0] = vec4(texture2D(A, x).x+incA*0.1,0,0,1);               \n"
"    gl_FragData[1] = vec4(texture2D(A, x).x+incB,0,0,1);                   \n"  
"}";
const char * glsl_codeB = ""
"#version 120\n"
"uniform sampler2D B;                                                       \n"
"uniform sampler2D C;                                                       \n"
"varying vec2 x; // texture coordinates                                     \n"
"uniform float dx; // delta                                                 \n"
"\n"        
"void main()                                                                \n"
"{                                                                          \n"
"    gl_FragData[0] = vec4(texture2D(B, x).x + texture2D(C, x).x, 0, 0, 1); \n"
"}";
//----------------------------------------------------------------------------//
const char * glsl_boilerplateA = ""
"#version 120\n"
"uniform sampler2D A;\n"
"uniform int incA;\n"
"uniform float incB;\n"
"varying vec2 x; // texture coordinates\n"
"uniform float dx; // delta\n"
"\n"        
"void main()\n"
"{\n"
"    // gl_FragData[0] is buffer B\n"
"    gl_FragData[0] = vec4(0,0,0,1);\n"
"\n"
"    // gl_FragData[1] is buffer C\n"
"    gl_FragData[1] = vec4(0,0,0,1);\n"
"}";
const char * glsl_boilerplateB = ""
"#version 120\n"
"uniform sampler2D B;\n"
"uniform sampler2D C;\n"
"varying vec2 x; // texture coordinates\n"
"uniform float dx; // delta\n"
"\n"        
"void main()\n"
"{\n"
"    // gl_FragData[0] is buffer A\n"
"    gl_FragData[0] = vec4(0,0,0,1);\n"
"}";
//----------------------------------------------------------------------------//
inline bool equal(float a, float b)
{
    return fabs(a-b) < 0.001;
}
//----------------------------------------------------------------------------//
void test(gpuip::GpuEnvironment env, const char * codeA, const char * codeB,
          const char * boilerplateA, const char * boilerplateB)
{
    if (!gpuip::ImageProcessor::CanCreate(env)) {
        return;
    }
    
    const char * gpu[3] = {"OpenCL", "CUDA", "GLSL"};
    std::cout << "Testing " << gpu[env] << "..." << std::endl;
    
    const unsigned int width = 4;
    const unsigned int height = 4;
    const unsigned int N = width * height;
    gpuip::ImageProcessor::Ptr ip(gpuip::ImageProcessor::Create(env));
    ip->SetDimensions(width, height);

    gpuip::Buffer::Ptr b1 = ip->CreateBuffer("b1", gpuip::Buffer::FLOAT, 1);
    gpuip::Buffer::Ptr b2 = ip->CreateBuffer("b2", gpuip::Buffer::FLOAT, 1);
    gpuip::Buffer::Ptr b3 = ip->CreateBuffer("b3", gpuip::Buffer::FLOAT, 1);

    gpuip::Kernel::Ptr kernelA = ip->CreateKernel("my_kernelA");
    assert(kernelA.get() != NULL);
    assert(kernelA->name == std::string("my_kernelA"));
    kernelA->code = codeA;
    kernelA->inBuffers.push_back(gpuip::Kernel::BufferLink(b1,"A"));
    kernelA->outBuffers.push_back(gpuip::Kernel::BufferLink(b2,"B"));
    kernelA->outBuffers.push_back(gpuip::Kernel::BufferLink(b3,"C"));


    const gpuip::Parameter<int> incA("incA", 2);
    const gpuip::Parameter<float> incB("incB", 0.25);
    kernelA->paramsInt.push_back(incA);
    kernelA->paramsFloat.push_back(incB);
    assert(ip->BoilerplateCode(kernelA) == std::string(boilerplateA));
    
    gpuip::Kernel::Ptr kernelB = ip->CreateKernel("my_kernelB");
    assert(kernelB.get() != NULL);
    assert(kernelB->name == std::string("my_kernelB"));
    kernelB->code = codeB;
    kernelB->inBuffers.push_back(gpuip::Kernel::BufferLink(b2,"B"));
    kernelB->inBuffers.push_back(gpuip::Kernel::BufferLink(b3,"C"));
    kernelB->outBuffers.push_back(gpuip::Kernel::BufferLink(b1,"A"));
    assert(ip->BoilerplateCode(kernelB) == std::string(boilerplateB));
    
    std::string err;
    std::cout << ip->Allocate(&err) << std::endl;
     std::cout << err << std::endl;
    //assert(ip->Allocate(&err) >= 0);
    //assert(ip->Allocate(&err) >= 0); //reiniting should not break things
    
    std::vector<float> data_in(N);
    for(size_t i = 0; i < data_in.size(); ++i) {
        data_in[i] = i;
    }
    std::cout << ip->Copy(b1, gpuip::Buffer::COPY_TO_GPU,
                   data_in.data(), &err)  << std::endl;
    std::cout << err << std::endl;

    std::cout << ip->Build(&err)  << std::endl;
    std::cout << ip->Build(&err)  << std::endl; // rebuilding should not break things
  
    std::cout << ip->Run(&err)  << std::endl;
    
    std::vector<float> data_outA(N), data_outB(N), data_outC(N);
    std::cout << ip->Copy(b1, gpuip::Buffer::COPY_FROM_GPU,data_outA.data(),&err)  << std::endl;
    std::cout << ip->Copy(b2, gpuip::Buffer::COPY_FROM_GPU,data_outB.data(),&err)  << std::endl;
    std::cout << ip->Copy(b3, gpuip::Buffer::COPY_FROM_GPU,data_outC.data(),&err)  << std::endl;

    for(unsigned int i = 0; i < N; ++i) {
        std::cout << data_outA[i] << " " << data_outB[i]  << " " << data_outC[i] << std::endl;
        // Check first kernel call, where B = A + 0.2, C = A + 0.25
        assert(equal(data_outB[i], data_in[i] + incA.value*0.1));
        assert(equal(data_outC[i], data_in[i] + incB.value));

        // Check second kernel call, where A = B + C
        assert(equal(data_outA[i], data_outB[i] + data_outC[i]));
    }
    std::cout << "Test passed!" << std::endl;
}
//----------------------------------------------------------------------------//
int main()
{
    test(gpuip::OpenCL, opencl_codeA, opencl_codeB,
         opencl_boilerplateA, opencl_boilerplateB);
    test(gpuip::CUDA, cuda_codeA, cuda_codeB,
         cuda_boilerplateA, cuda_boilerplateB);
    test(gpuip::GLSL, glsl_codeA, glsl_codeB,
         glsl_boilerplateA, glsl_boilerplateB);
    return 0;
}
//----------------------------------------------------------------------------//
