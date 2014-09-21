#include <gpuip.h>
#include <stdlib.h>
#include <math.h>
//----------------------------------------------------------------------------//
const char * cuda_code = ""
"__global__ void                                                             \n"
"my_kernel(float * B,                                                        \n"
"          const int width,                                                  \n"
"          const int height)                                                 \n"
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
"    B[idx] = tex2D(A, (x+0.5)/width, (y+0.5)/height);                       \n"
"}";
const char * cuda_boilerplate = "";
//----------------------------------------------------------------------------//
std::string err;
//----------------------------------------------------------------------------//
inline void correct(bool statement, const char * msg)
{
    if(!statement) {
        throw std::runtime_error(std::string(msg));
    }
}
//----------------------------------------------------------------------------//
inline void correct(double execution_time)
{
    correct(execution_time >= 0, err.c_str());
}
//----------------------------------------------------------------------------//
inline bool correct(float a, float b, const char * msg)
{
    correct(fabs(a-b) < 0.001, msg);
}
//----------------------------------------------------------------------------//
void test(gpuip::GpuEnvironment env,
          const char * code,
          const char * boilerplate)
{
    if (!gpuip::ImageProcessor::CanCreate(env)) {
        return;
    }

    const char * gpu[3] = {"OpenCL", "CUDA", "GLSL"};
    std::cout << "Testing " << gpu[env] << "..." << std::endl;
    gpuip::ImageProcessor::Ptr ip(gpuip::ImageProcessor::Create(env));
    
    const unsigned int width = 16;
    const unsigned int height = 17;
    const unsigned int N = width * height;

    gpuip::Buffer::Ptr b1 = ip->CreateBuffer("b1", gpuip::Buffer::FLOAT,
                                             2*width, 2*height, 1);
    b1->isTexture = true;
    gpuip::Buffer::Ptr b2 = ip->CreateBuffer("b2", gpuip::Buffer::FLOAT,
                                             width, height, 1);

    gpuip::Kernel::Ptr kernel = ip->CreateKernel("my_kernel");
    kernel->code = code;
    kernel->inBuffers.push_back(gpuip::Kernel::BufferLink(b1,"A"));
    kernel->outBuffers.push_back(gpuip::Kernel::BufferLink(b2,"B"));

    correct(ip->Build(&err));
    std::cout << ip->BoilerplateCode(kernel) << std::endl;
    correct(ip->Allocate(&err) >= 0, err.c_str());

    std::vector<float> data_in(2*width*2*height);
    for(size_t i = 0; i < 2 * width; ++i) {
        for(size_t j = 0; j < 2 * height; ++j) {
            data_in[i + 2* width * j] = i + 2* width * j;
        }
    }
   
    correct(ip->Copy(b1, gpuip::Buffer::COPY_TO_GPU,data_in.data(), &err));
    correct(ip->Run(&err));

    std::vector<float> data_out(width*height);
    correct(ip->Copy(b2, gpuip::Buffer::COPY_FROM_GPU,data_out.data(), &err));
  
    for(size_t i = 0; i < width; ++i) {
        for(size_t j = 0; j < height; ++j) {
            int ii = 2*i;
            int jj = 2*j;
            int ww = 2*width;
            correct(data_out[i + width * j],
                    0.25*(data_in[2*i+ww*2*j]+ data_in[2*i+1+ww*2*j]+
                          data_in[2*i+ww*(2*j+1)]+ data_in[2*i+1+ww*(2*j+1)]),
                    "lerp");
        }
    }
     std::cout << "Test passed!" << std::endl;
}
//----------------------------------------------------------------------------//  
int main()
{
    //test(gpuip::OpenCL, opencl_codeA, opencl_codeB,
    //     opencl_boilerplateA, opencl_boilerplateB);
    test(gpuip::CUDA, cuda_code, cuda_boilerplate);
    //test(gpuip::GLSL, glsl_codeA, glsl_codeB,
    //     glsl_boilerplateA, glsl_boilerplateB);
    return 0;
}
//----------------------------------------------------------------------------//
