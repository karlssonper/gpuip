#include <gpuip.h>
#include <OpenEXR/ImfRgbaFile.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <cassert>
#include <ctime>
#ifndef NUM_TESTS
#  define NUM_TESTS 5
#endif
#define ALPHA 0.5
#define BLUR_N 4

inline std::string GetKernelCode(const std::string & filename)
{
    std::stringstream buffer;
    buffer << std::ifstream(filename.c_str()).rdbuf();
    return buffer.str();
}

inline void GetImage(const char * filename,
                     std::vector<Imf::Rgba> & data,
                     unsigned int & width,
                     unsigned int & height)
{
    using namespace Imf;
    RgbaInputFile file(filename);
    Imath::Box2i dw = file.dataWindow();
    width = dw.max.x - dw.min.x + 1;
    height = dw.max.y - dw.min.y + 1;
    data.resize(width*height);
    file.setFrameBuffer(data.data(), 1, width);
    file.readPixels(dw.min.y, dw.max.y);
}

std::clock_t TimerStart()
{
    return std::clock();
}

long double TimerStop(std::clock_t start)
{
    return ( std::clock() - start ) / (long double) CLOCKS_PER_SEC;
};

#ifdef _GPUIP_TEST_WITH_OPENMP
#include <omp.h>
std::clock_t TimerStartOMP()
{
    return omp_get_wtime();
}
long double TimerStopOMP(std::clock_t start)
{
    return omp_get_wtime() - start;
};
#endif

void LerpCPU(const std::vector<Imf::Rgba> & A,
             const std::vector<Imf::Rgba> & B,
             std::vector<Imf::Rgba> & C)
{
    double time = 0;
    for (int test = 0; test < NUM_TESTS; ++test) {
        std::clock_t start = TimerStart();
        for(size_t i = 0; i < A.size(); ++i) {
            C[i].r = (1-ALPHA) * A[i].r + ALPHA * B[i].r;
            C[i].g = (1-ALPHA) * A[i].g + ALPHA * B[i].g;
            C[i].b = (1-ALPHA) * A[i].b + ALPHA * B[i].b;
            C[i].a = (1-ALPHA) * A[i].a + ALPHA * B[i].a;
        }
        time += double(TimerStop(start)*1000.0);
    }
    printf("CPU:    %.1lf ms.\n", time/NUM_TESTS);
}

void LerpCPUMultiThreaded(const std::vector<Imf::Rgba> & A,
                          const std::vector<Imf::Rgba> & B,
                          std::vector<Imf::Rgba> & C,
                          const unsigned int width,
                          const unsigned int height)
{
#ifndef _GPUIP_TEST_WITH_OPENMP
    return;
#endif
    double time = 0;
    for (int test = 0; test < NUM_TESTS; ++test) {
        std::clock_t start = TimerStartOMP();
#pragma omp parallel for
        for(unsigned int y = 0; y < height; ++y) {
            for (unsigned int x = 0; x < width; ++x) {
                const size_t idx = x + width * y;
                C[idx].r = (1-ALPHA) * A[idx].r + ALPHA * B[idx].r;
                C[idx].g = (1-ALPHA) * A[idx].g + ALPHA * B[idx].g;
                C[idx].b = (1-ALPHA) * A[idx].b + ALPHA * B[idx].b;
                C[idx].a = (1-ALPHA) * A[idx].a + ALPHA * B[idx].a;
            }
        }
        time += double(TimerStopOMP(start)*1000.0);
    }
    printf("CPU MT: %.1lf ms.\n", time/NUM_TESTS);
}

void BoxBlurCPU(const std::vector<Imf::Rgba> & A,
                std::vector<Imf::Rgba> & B,
                unsigned int width,
                unsigned int height)
{
    double time = 0;
    for (int test = 0; test < NUM_TESTS; ++test) {
        std::clock_t start = TimerStart();
        for(unsigned int y = 0; y < height; ++y) {
            for (unsigned int x = 0; x < width; ++x) {
                Imf::Rgba out;
                int count = 0;
                for(int j = y - BLUR_N; j <= y + BLUR_N; ++j)  {
                    for(int i = x - BLUR_N; i <= x+BLUR_N; ++i) {
                        if (i>=0 && j>= 0 && i < width && j < height) {
                            out.r += A[i + width * j].r;
                            out.g += A[i + width * j].g;
                            out.b += A[i + width * j].b;
                            out.a += A[i + width * j].a;
                            count += 1;
                        }
                    }
                }
                B[x + width * y].r = out.r / count;
                B[x + width * y].g = out.g / count;
                B[x + width * y].b = out.b / count;
                B[x + width * y].a = out.a / count;
            }
        }
        time += double(TimerStop(start)*1000.0);
    }
    printf("CPU:    %.1lf ms.\n", time/NUM_TESTS);
}

void BoxBlurCPUMultiThreaded(const std::vector<Imf::Rgba> & A,
                             std::vector<Imf::Rgba> & B,
                             const unsigned int width,
                             const unsigned int height)
{
#ifndef _GPUIP_TEST_WITH_OPENMP
    return;
#endif
    double time = 0;
    for (int test = 0; test < NUM_TESTS; ++test) {
        std::clock_t start = TimerStartOMP();
    
#pragma omp parallel for
        for(unsigned int y = 0; y < height; ++y) {
            for (unsigned int x = 0; x < width; ++x) {
                Imf::Rgba out;
                int count = 0;
                for(int j = y - BLUR_N; j <= y + BLUR_N; ++j)  {
                    for(int i = x - BLUR_N; i <= x+BLUR_N; ++i) {
                        if (i>=0 && j>= 0 && i < width && j < height) {
                            out.r += A[i + width * j].r;
                            out.g += A[i + width * j].g;
                            out.b += A[i + width * j].b;
                            out.a += A[i + width * j].a;
                            count += 1;
                        }
                    }
                }
                B[x + width * y].r = out.r / count;
                B[x + width * y].g = out.g / count;
                B[x + width * y].b = out.b / count;
                B[x + width * y].a = out.a / count;
            }
        }
        time += double(TimerStopOMP(start)*1000.0);
    }
    printf("CPU MT: %.1lf ms.\n", time/NUM_TESTS);
}

void GaussianBlurCPU(const std::vector<Imf::Rgba> & A,
                     std::vector<Imf::Rgba> & B,
                     unsigned int width,
                     unsigned int height)
{
    double time = 0;
    for (int test = 0; test < NUM_TESTS; ++test) {
        std::clock_t start = TimerStart();
        for(unsigned int y = 0; y < height; ++y) {
            for (unsigned int x = 0; x < width; ++x) {
                Imf::Rgba out;
                float totWeight = 0;
                const float invdx2 = 1.0/(width*width);
                float w;
                for(int j = y - BLUR_N; j <= y + BLUR_N; ++j)  {
                    for(int i = x - BLUR_N; i <= x+BLUR_N; ++i) {
                        if (i>=0 && j>= 0 && i < width && j < height) {
                            w = exp(-invdx2*((i-x)*(i-x) + (j-y)*(j-y)));
                            out.r += w*A[i + width * j].r;
                            out.g += w*A[i + width * j].g;
                            out.b += w*A[i + width * j].b;
                            out.a += w*A[i + width * j].a;
                            totWeight += w;
                        }
                    }
                }
                B[x + width * y].r = out.r / totWeight;
                B[x + width * y].g = out.g / totWeight;
                B[x + width * y].b = out.b / totWeight;
                B[x + width * y].a = out.a / totWeight;
            }
        }
        time += double(TimerStop(start)*1000.0);
    }
    printf("CPU:    %.1lf ms.\n", time/NUM_TESTS);
}

void GaussianBlurCPUMultiThreaded(const std::vector<Imf::Rgba> & A,
                                  std::vector<Imf::Rgba> & B,
                                  const unsigned int width,
                                  const unsigned int height)
{
#ifndef _GPUIP_TEST_WITH_OPENMP
    return;
#endif
    double time = 0;
    for (int test = 0; test < NUM_TESTS; ++test) {
        std::clock_t start = TimerStartOMP();
#pragma omp parallel for
        for(unsigned int y = 0; y < height; ++y) {
            for (unsigned int x = 0; x < width; ++x) {
                Imf::Rgba out;
                float totWeight = 0;
                const float invdx2 = 1.0/(width*width);
                float w;
                for(int j = y - BLUR_N; j <= y + BLUR_N; ++j)  {
                    for(int i = x - BLUR_N; i <= x+BLUR_N; ++i) {
                        if (i>=0 && j>= 0 && i < width && j < height) {
                            w = exp(-invdx2*((i-x)*(i-x) + (j-y)*(j-y)));
                            out.r += w*A[i + width * j].r;
                            out.g += w*A[i + width * j].g;
                            out.b += w*A[i + width * j].b;
                            out.a += w*A[i + width * j].a;
                            totWeight += w;
                        }
                    }
                }
                B[x + width * y].r = out.r / totWeight;
                B[x + width * y].g = out.g / totWeight;
                B[x + width * y].b = out.b / totWeight;
                B[x + width * y].a = out.a / totWeight;
            }
        }
        time =+ double(TimerStopOMP(start)*1000.0);
    }
    
    printf("CPU MT: %.1lf ms.\n", time/NUM_TESTS);
}

void GaussianBlurSeparableCPU(const std::vector<Imf::Rgba> & A,
                              std::vector<Imf::Rgba> & B,
                              std::vector<Imf::Rgba> & C,
                              unsigned int width,
                              unsigned int height)
{
    double time = 0;
    for (int test = 0; test < NUM_TESTS; ++test) {
        std::clock_t start = TimerStart();
        for(unsigned int y = 0; y < height; ++y) {
            for (unsigned int x = 0; x < width; ++x) {
                Imf::Rgba out;
                float totWeight = 0;
                const float invdx2 = 1.0/(width*width);
                float w;
                for(int i = x - BLUR_N; i <= x+BLUR_N; ++i) {
                    if (i>=0 &&  i < width) {
                        exp(-invdx2*((i-x)*(i-x)));
                        out.r += w*A[i + width * y].r;
                        out.g += w*A[i + width * y].g;
                        out.b += w*A[i + width * y].b;
                        out.a += w*A[i + width * y].a;
                        totWeight += w;
                    }
                }
                B[x + width * y].r = out.r / totWeight;
                B[x + width * y].g = out.g / totWeight;
                B[x + width * y].b = out.b / totWeight;
                B[x + width * y].a = out.a / totWeight;
            }
        }
        for(unsigned int y = 0; y < height; ++y) {
            for (unsigned int x = 0; x < width; ++x) {
                Imf::Rgba out;
                float totWeight = 0;
                const float invdx2 = 1.0/(width*width);
                float w;
                for(int j = y - BLUR_N; j <= y+BLUR_N; ++j) {
                    if (j>=0 &&  j < height) {
                        w = exp(-invdx2*((j-y)*(j-y)));
                        out.r += w*B[x + width * j].r;
                        out.g += w*B[x + width * j].g;
                        out.b += w*B[x + width * j].b;
                        out.a += w*B[x + width * j].a;
                        totWeight += w;
                    }
                }
                C[x + width * y].r = out.r / totWeight;
                C[x + width * y].g = out.g / totWeight;
                C[x + width * y].b = out.b / totWeight;
                C[x + width * y].a = out.a / totWeight;
            }
        }
        time += double(TimerStop(start)*1000.0);
    }
    printf("CPU:    %.1lf ms.\n", time/NUM_TESTS);
}

void GaussianBlurSeparableCPUMultiThreaded(const std::vector<Imf::Rgba> & A,
                                           std::vector<Imf::Rgba> & B,
                                           std::vector<Imf::Rgba> & C,
                                           unsigned int width,
                                           unsigned int height)
{
#ifndef _GPUIP_TEST_WITH_OPENMP
    return;
#endif
    double time = 0;
    for (int test = 0; test < NUM_TESTS; ++test) {
        std::clock_t start = TimerStartOMP();
#pragma omp parallel for
        for(unsigned int y = 0; y < height; ++y) {
            for (unsigned int x = 0; x < width; ++x) {
                Imf::Rgba out;
                float totWeight = 0;
                const float invdx2 = 1.0/(width*width);
                float w;
                for(int i = x - BLUR_N; i <= x+BLUR_N; ++i) {
                    if (i>=0 &&  i < width) {
                        exp(-invdx2*((i-x)*(i-x)));
                        out.r += w*A[i + width * y].r;
                        out.g += w*A[i + width * y].g;
                        out.b += w*A[i + width * y].b;
                        out.a += w*A[i + width * y].a;
                        totWeight += w;
                    }
                }
                B[x + width * y].r = out.r / totWeight;
                B[x + width * y].g = out.g / totWeight;
                B[x + width * y].b = out.b / totWeight;
                B[x + width * y].a = out.a / totWeight;
            }
        }
#pragma omp parallel for
        for(unsigned int y = 0; y < height; ++y) {
            for (unsigned int x = 0; x < width; ++x) {
                Imf::Rgba out;
                float totWeight = 0;
                const float invdx2 = 1.0/(width*width);
                float w;
                for(int j = y - BLUR_N; j <= y+BLUR_N; ++j) {
                    if (j>=0 &&  j < height) {
                        w = exp(-invdx2*((j-y)*(j-y)));
                        out.r += w*B[x + width * j].r;
                        out.g += w*B[x + width * j].g;
                        out.b += w*B[x + width * j].b;
                        out.a += w*B[x + width * j].a;
                        totWeight += w;
                    }
                }
                C[x + width * y].r = out.r / totWeight;
                C[x + width * y].g = out.g / totWeight;
                C[x + width * y].b = out.b / totWeight;
                C[x + width * y].a = out.a / totWeight;
            }
        }
        time += double(TimerStopOMP(start)*1000.0);
    }
    printf("CPU MT: %.1lf ms.\n", time/NUM_TESTS);
}

void GaussianBlurSeparableGPU(gpuip::GpuEnvironment env,
                              const std::string & codeHor,
                              const std::string & codeVert,
                              unsigned int width,
                              unsigned int height,
                              std::vector<Imf::Rgba> & A,
                              std::vector<Imf::Rgba> & C)
{
    if (!gpuip::ImageProcessor::CanCreate(env)) {
        return;
    }

    gpuip::ImageProcessor::Ptr ip(gpuip::ImageProcessor::Create(env));
    ip->SetDimensions(width, height);

    gpuip::Buffer::Ptr b1 = ip->CreateBuffer("b1", gpuip::Buffer::HALF, 4);
    gpuip::Buffer::Ptr b2 = ip->CreateBuffer("b2", gpuip::Buffer::HALF, 4);
    gpuip::Buffer::Ptr b3 = ip->CreateBuffer("b3", gpuip::Buffer::HALF, 4);

    gpuip::Kernel::Ptr kernelA = ip->CreateKernel("gaussian_blur_hor");
    kernelA->code = codeHor;
    kernelA->inBuffers.push_back(gpuip::Kernel::BufferLink(b1,"input"));
    kernelA->outBuffers.push_back(gpuip::Kernel::BufferLink(b2,"output"));

    const gpuip::Parameter<int> nA("n", BLUR_N);
    kernelA->paramsInt.push_back(nA);

    gpuip::Kernel::Ptr kernelB = ip->CreateKernel("gaussian_blur_vert");
    kernelB->code = codeVert;
    kernelB->inBuffers.push_back(gpuip::Kernel::BufferLink(b2,"input"));
    kernelB->outBuffers.push_back(gpuip::Kernel::BufferLink(b3,"output"));

    const gpuip::Parameter<int> nB("n", BLUR_N);
    kernelB->paramsInt.push_back(nB);

    std::string error;
    ip->Allocate(&error);
    ip->Build(&error);

    double tt = 0;
    double t = 0;
    for (int test = 0; test < NUM_TESTS; ++test) {
        tt += ip->Copy(b1, gpuip::Buffer::COPY_TO_GPU, A.data(), &error);
        t += ip->Run(&error);
        tt += ip->Copy(b3, gpuip::Buffer::COPY_FROM_GPU, C.data(), &error);
    }
    tt /= NUM_TESTS;
    t /= NUM_TESTS;
    const char * gpu[3] = {"OpenCL:", "CUDA:  ", "GLSL:  " };
    printf("%s %.1lf ms, Process %.1lf ms (%.1lf%%), Copy %.1lf ms (%.1lf%%)\n",
           gpu[env], tt+t,t,  t/(tt+t)*100, tt, tt/(tt+t)*100);
}

void BlurGPU(gpuip::GpuEnvironment env,
             const std::string & code,
             const std::string & blur,
             unsigned int width,
             unsigned int height,
             std::vector<Imf::Rgba> & A,
             std::vector<Imf::Rgba> & B)
{
    if (!gpuip::ImageProcessor::CanCreate(env)) {
        return;
    }

    gpuip::ImageProcessor::Ptr ip(gpuip::ImageProcessor::Create(env));
    ip->SetDimensions(width, height);

    gpuip::Buffer::Ptr b1 = ip->CreateBuffer("b1", gpuip::Buffer::HALF, 4);
    gpuip::Buffer::Ptr b2 = ip->CreateBuffer("b2", gpuip::Buffer::HALF, 4);
    gpuip::Buffer::Ptr b3 = ip->CreateBuffer("b3", gpuip::Buffer::HALF, 4);

    gpuip::Kernel::Ptr kernel = ip->CreateKernel(blur);
    kernel->code = code;
    kernel->inBuffers.push_back(gpuip::Kernel::BufferLink(b1,"input"));
    kernel->outBuffers.push_back(gpuip::Kernel::BufferLink(b2,"output"));

    const gpuip::Parameter<int> n("n", BLUR_N);
    kernel->paramsInt.push_back(n);

    std::string error;
    ip->Allocate(&error);
    ip->Build(&error);

    double tt = 0;
    double t = 0;
    for (int test = 0; test < NUM_TESTS; ++test) {
        tt += ip->Copy(b1, gpuip::Buffer::COPY_TO_GPU, A.data(), &error);
        t += ip->Run(&error);
        tt += ip->Copy(b2, gpuip::Buffer::COPY_FROM_GPU, B.data(), &error);
    }
    tt /= NUM_TESTS;
    t /= NUM_TESTS;
    const char * gpu[3] = {"OpenCL:", "CUDA:  ", "GLSL:  " };
    printf("%s %.1lf ms, Process %.1lf ms (%.1lf%%), Copy %.1lf ms (%.1lf%%)\n",
           gpu[env], tt+t,t,  t/(tt+t)*100, tt, tt/(tt+t)*100);
}

void LerpGPU(gpuip::GpuEnvironment env,
             const std::string & code,
             unsigned int width,
             unsigned int height,
             std::vector<Imf::Rgba> & A,
             std::vector<Imf::Rgba> & B,
             std::vector<Imf::Rgba> & C)
{
    if (!gpuip::ImageProcessor::CanCreate(env)) {
        return;
    }

    gpuip::ImageProcessor::Ptr ip(gpuip::ImageProcessor::Create(env));
    ip->SetDimensions(width, height);

    gpuip::Buffer::Ptr b1 = ip->CreateBuffer("b1", gpuip::Buffer::HALF, 4);
    gpuip::Buffer::Ptr b2 = ip->CreateBuffer("b2", gpuip::Buffer::HALF, 4);
    gpuip::Buffer::Ptr b3 = ip->CreateBuffer("b3", gpuip::Buffer::HALF, 4);    
    
    gpuip::Kernel::Ptr kernel = ip->CreateKernel("lerp");
    kernel->code = code;
    kernel->inBuffers.push_back(gpuip::Kernel::BufferLink(b1,"a"));
    kernel->outBuffers.push_back(gpuip::Kernel::BufferLink(b2,"b"));
    kernel->outBuffers.push_back(gpuip::Kernel::BufferLink(b3,"c"));
   
    const gpuip::Parameter<float> alpha( "alpha", ALPHA);
    kernel->paramsFloat.push_back(alpha);

    std::string error;
    ip->Allocate(&error);
    ip->Build(&error);
    
    double tt = 0;
    double t = 0;
    for (int test = 0; test < NUM_TESTS; ++test) {
        tt += ip->Copy(b1, gpuip::Buffer::COPY_TO_GPU, A.data(), &error);
        tt += ip->Copy(b2, gpuip::Buffer::COPY_TO_GPU, B.data(), &error);
        t += ip->Run(&error);
        tt += ip->Copy(b3, gpuip::Buffer::COPY_FROM_GPU, C.data(), &error);
    }
    tt /= NUM_TESTS;
    t /= NUM_TESTS;
    const char * gpu[3] = {"OpenCL:", "CUDA:  ", "GLSL:  " };
    printf("%s %.1lf ms, Process %.1lf ms (%.1lf%%), Copy %.1lf ms (%.1lf%%)\n",
           gpu[env], tt+t,t,  t/(tt+t)*100, tt, tt/(tt+t)*100);
}

int main(int argc, char ** argv)
{
    if (argc != 3) {
        std::cerr << "usage: ./test_speed path/to/exr/image path/to/kernels\n";
        return 1;
    }

    // Read exr image from file
    std::vector<Imf::Rgba> dataA, dataB, dataCPU, dataGPU;
    unsigned int width, height;
    GetImage(argv[1], dataA, width, height);
    dataB.resize(dataA.size());
    dataCPU.resize(dataA.size());
    dataGPU.resize(dataA.size());
    
    // dataB is dataA reversed
    for(size_t i = 0; i < dataA.size(); ++i) {
        dataB[dataB.size()-1-i] = dataA[i]; 
    }

    std::string kernels_dir(argv[2]);
    printf("---------------------------------------------------------------\n"
           "|                  LERP                                       |\n"
           "---------------------------------------------------------------\n");
    LerpCPU(dataA, dataB, dataCPU);
    LerpCPUMultiThreaded(dataA, dataB, dataCPU, width, height);
    LerpGPU(gpuip::OpenCL,
            GetKernelCode(kernels_dir + "lerp.cl").c_str(),
            width, height, dataA, dataB, dataGPU);
    LerpGPU(gpuip::CUDA,
            GetKernelCode(kernels_dir + "lerp.cu").c_str(),
            width, height, dataA, dataB, dataGPU);
    LerpGPU(gpuip::GLSL, GetKernelCode(kernels_dir + "lerp.glsl").c_str(),
            width, height,dataA, dataB, dataGPU);

    printf("---------------------------------------------------------------\n"
           "|                  BOX BLUR                                   |\n"
           "---------------------------------------------------------------\n");
    BoxBlurCPU(dataA, dataCPU, width, height);
    BoxBlurCPUMultiThreaded(dataA, dataCPU, width, height);
    BlurGPU(gpuip::OpenCL,
            GetKernelCode(kernels_dir + "box_blur.cl").c_str(),
            "box_blur", width, height, dataA, dataGPU);
    BlurGPU(gpuip::CUDA, GetKernelCode(kernels_dir + "box_blur.cu").c_str(),
            "box_blur", width, height, dataA, dataGPU);
    BlurGPU(gpuip::GLSL,GetKernelCode(kernels_dir + "box_blur.glsl").c_str(),
            "box_blur",  width, height, dataA, dataGPU);

    printf("---------------------------------------------------------------\n"
           "|                  GAUSSIAN BLUR                              |\n"
           "---------------------------------------------------------------\n");
    GaussianBlurCPU(dataA, dataCPU, width, height);
    GaussianBlurCPUMultiThreaded(dataA, dataCPU, width, height);
    BlurGPU(gpuip::OpenCL,
            GetKernelCode(kernels_dir + "gaussian_blur.cl").c_str(),
            "gaussian_blur", width, height, dataA, dataGPU);
    BlurGPU(gpuip::CUDA,
            GetKernelCode(kernels_dir + "gaussian_blur.cu").c_str(),
            "gaussian_blur", width, height, dataA, dataGPU);
    BlurGPU(gpuip::GLSL,
            GetKernelCode(kernels_dir + "gaussian_blur.glsl").c_str(),
            "gaussian_blur",  width, height, dataA, dataGPU);

    printf("---------------------------------------------------------------\n"
           "|                  SEPARABLE GAUSSIAN BLUR                    |\n"
           "---------------------------------------------------------------\n");
    GaussianBlurSeparableCPU(dataA, dataB, dataCPU, width, height);
    GaussianBlurSeparableCPUMultiThreaded(dataA, dataB, dataCPU, width, height);
    GaussianBlurSeparableGPU(
        gpuip::OpenCL,
        GetKernelCode(kernels_dir + "gaussian_blur_hor.cl").c_str(),
        GetKernelCode(kernels_dir + "gaussian_blur_vert.cl").c_str(),
        width, height, dataA, dataGPU);
    GaussianBlurSeparableGPU(
        gpuip::CUDA,
        GetKernelCode(kernels_dir + "gaussian_blur_hor.cu").c_str(),
        GetKernelCode(kernels_dir + "gaussian_blur_vert.cu").c_str(),
        width, height, dataA, dataGPU);
    GaussianBlurSeparableGPU(
        gpuip::GLSL,
        GetKernelCode(kernels_dir + "gaussian_blur_hor.glsl").c_str(),
        GetKernelCode(kernels_dir + "gaussian_blur_vert.glsl").c_str(),
        width, height, dataA, dataGPU);
    return 0;
}
