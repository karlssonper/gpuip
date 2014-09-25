/*
The MIT License (MIT)

Copyright (c) 2014 Per Karlsson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "cuda.h"
#include "cuda_error.h"
#include "helper_math.cuh"
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
//----------------------------------------------------------------------------//
// Plugin interface
extern "C" GPUIP_DECLSPEC gpuip::ImplInterface * CreateImpl()
{
    return new gpuip::CUDAImpl();
}
extern "C" GPUIP_DECLSPEC void DeleteImpl(gpuip::ImplInterface * impl)
{
    delete impl;
}
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
inline int _cudaGetMaxGflopsDeviceId();
//----------------------------------------------------------------------------//
inline std::string _GetTypeStr(const Buffer::Ptr & buffer);
//----------------------------------------------------------------------------//
inline int _removeFile(const char * filename);
//----------------------------------------------------------------------------//
int _execPipe(const char* cmd, std::string * err);
//----------------------------------------------------------------------------//
CUresult _SetBufferArgs(const CUfunction & cudaKernel,
                        const std::vector<Kernel::BufferLink> & buffers,
                        std::map<std::string, float*> & cudaBuffers,
                        int & paramOffset);
//----------------------------------------------------------------------------//
template<typename T>
CUresult _SetParamArgs(const CUfunction & cudaKernel,
                       const std::vector<Parameter<T> > & params,
                       int & paramOffset);
//----------------------------------------------------------------------------//
void _BoilerplateBufferArgs(
    std::stringstream & ss,
    const std::vector<Kernel::BufferLink> & bufferLinks,
    const std::string & indent,
    int & argcount,
    bool input);
//----------------------------------------------------------------------------//
template<typename T>
void _BoilerplateParamArgs(std::stringstream & ss,
                           const std::vector<Parameter<T > > & params,
                           const char * typenameStr,
                           const std::string & indent,
                           int & argcount);
//----------------------------------------------------------------------------//
void _BoilerplateHalfToFloat(
    std::stringstream & ss,
    const std::vector<Kernel::BufferLink> & bufferLinks);
//----------------------------------------------------------------------------//
void _BoilerplateKernelCode(
    std::stringstream & ss,
    const std::vector<Kernel::BufferLink> & bufferLinks);
//----------------------------------------------------------------------------//
void _BoilerplateFloatToHalf(
    std::stringstream & ss,
    const std::vector<Kernel::BufferLink> & bufferLinks);
//----------------------------------------------------------------------------//
CUDAImpl::CUDAImpl()
        : _cudaBuild(false)
{
    if (cudaSetDevice(_cudaGetMaxGflopsDeviceId()) != cudaSuccess) {
        throw std::logic_error("gpuip::CUDAImpl() could not set device id");
    };
    cudaFree(0); //use runtime api to create a CUDA context implicitly

    cudaEventCreate(&_start);
    cudaEventCreate(&_stop);
}
//----------------------------------------------------------------------------//
CUDAImpl::~CUDAImpl()
{
    std::string err;
    if(!_FreeBuffers(&err)) {
        std::cerr << err << std::endl;
    }

    err.clear();
    if(!_UnloadModule(&err)) {
        std::cerr << err << std::endl;
    }
}
//----------------------------------------------------------------------------//
double CUDAImpl::Allocate(std::string * err)
{
    _StartTimer();

    if (!_FreeBuffers(err)) {
        return GPUIP_ERROR;
    }
    
    std::map<std::string,Buffer::Ptr>::const_iterator it;
    for(it = _buffers.begin(); it != _buffers.end(); ++it) {
        _cudaBuffers[it->second->Name()] = NULL;
        cudaError_t c_err = cudaMalloc(&_cudaBuffers[it->second->Name()],
                                       _BufferSize(it->second));
        if(_cudaErrorMalloc(c_err, err)) {
            return GPUIP_ERROR;
        }
    }
    return _StopTimer();
}
//----------------------------------------------------------------------------//
double CUDAImpl::Build(std::string * err)
{
    _StartTimer();

    if(!_UnloadModule(err)) {
        return GPUIP_ERROR;
    }

    const char * file_helper_math_h = ".helper_math.h";
    const char * file_temp_cu = ".temp.cu";
    const char * file_temp_ptx = ".temp.ptx";
    
    // Includes vector float operations such as mult, add etc
    std::ofstream out_helper(file_helper_math_h);
    out_helper << get_cuda_helper_math();
    out_helper.close();
    
    // Create temporary file to compile
    std::ofstream out(file_temp_cu);
    out << "#include \"" << file_helper_math_h << "\"\n";
    out << "extern \"C\" { \n"; // To avoid function name mangling 
    for(size_t i = 0; i < _kernels.size(); ++i) {
        out << _kernels[i]->Code() << "\n";
    }
    out << "}"; // End the extern C bracket
    out.close();

    std::stringstream ss;
    const char * cuda_bin_path = getenv("CUDA_BIN_PATH");
    if (cuda_bin_path  != NULL) {
        ss << cuda_bin_path << "/nvcc";
    } else {
        ss << "nvcc";
    }
    ss << " -ptx " << file_temp_cu << " -o " << file_temp_ptx
       << " --Wno-deprecated-gpu-targets"
       << " -include " << file_helper_math_h;
    if(sizeof(void *) == 4) {
        ss << " -m32";
    } else {
        ss << " -m64";
    }
#ifdef _WIN32
    const char * cl_bin_path = getenv("CL_BIN_PATH");
    if (cl_bin_path != NULL) {
        ss << " -ccbin \"" << cl_bin_path << "\"";
    }
#endif
    ss << " 2>&1" << std::endl; // get both standard output and error
    std::string pipe_err;
    int nvcc_exit_status = _execPipe(ss.str().c_str(), &pipe_err);

    // Cleanup temp text file
    _removeFile(file_helper_math_h);
    _removeFile(file_temp_cu);
        
    if (nvcc_exit_status) {
        (*err) = "Cuda error: Could not compile kernels:\n";
        (*err) += pipe_err;
        return GPUIP_ERROR;
    }

    // Load cuda ptx from file
    CUresult c_err = cuModuleLoad(&_cudaModule, ".temp.ptx");
    _removeFile(file_temp_ptx);
    if (_cudaErrorLoadModule(c_err, err)) {
        return GPUIP_ERROR;
    }

    _cudaKernels.resize(_kernels.size());
    for(size_t i = 0; i < _kernels.size(); ++i) {
        c_err = cuModuleGetFunction(&_cudaKernels[i], _cudaModule,
                                    _kernels[i]->Name().c_str());
        if (_cudaErrorGetFunction(c_err, err, _kernels[i]->Name())) {
            return GPUIP_ERROR;
        }
    }

    _cudaBuild = true;
    
    return _StopTimer();
}
//----------------------------------------------------------------------------//
double CUDAImpl::Run(std::string * err)
{
    _StartTimer();
    for(size_t i = 0; i < _kernels.size(); ++i) {
        if (!_LaunchKernel(*_kernels[i].get(), _cudaKernels[i], err)) {
            return GPUIP_ERROR;
        }
    }
    cudaDeviceSynchronize();
    return  _StopTimer();
}
//----------------------------------------------------------------------------//
double CUDAImpl::Copy(Buffer::Ptr buffer,
                      Buffer::CopyOperation op,
                      void * data,
                      std::string * err)
{
    _StartTimer();
    cudaError_t e = cudaSuccess;
    const size_t size = _BufferSize(buffer);
    if (op == Buffer::COPY_FROM_GPU) {
        e =cudaMemcpy(data, _cudaBuffers[buffer->Name()],
                      size, cudaMemcpyDeviceToHost);
    } else if (op == Buffer::COPY_TO_GPU) {
        e = cudaMemcpy(_cudaBuffers[buffer->Name()],data,
                       size, cudaMemcpyHostToDevice);
    }
    if (_cudaErrorCopy(e, err, buffer->Name(), op)) {
        return GPUIP_ERROR;
    }
    return _StopTimer();
}
//----------------------------------------------------------------------------//
bool CUDAImpl::_LaunchKernel(Kernel & kernel,
                             const CUfunction & cudaKernel,
                             std::string * err)
{
    // Set CUDA kernel arguments
    CUresult c_err;
    int paramOffset = 0;
    _SetBufferArgs(cudaKernel, kernel.InputBuffers(),
                   _cudaBuffers, paramOffset);
    _SetBufferArgs(cudaKernel, kernel.OutputBuffers(),
                   _cudaBuffers, paramOffset);
    _SetParamArgs(cudaKernel, kernel.ParamsInt(), paramOffset);
    _SetParamArgs(cudaKernel, kernel.ParamsFloat(), paramOffset);

    // int and width parameters
    c_err = cuParamSetv(cudaKernel, paramOffset, &_w, sizeof(int));
    paramOffset += sizeof(int);
    c_err = cuParamSetv(cudaKernel, paramOffset, &_h, sizeof(int));
    paramOffset += sizeof(int);
    
    // It should be fine to check once all the arguments have been set
    if(_cudaErrorCheckParamSet(c_err, err, kernel.Name())) {
        return false;
    }
    
    c_err = cuParamSetSize(cudaKernel, paramOffset);
    if (_cudaErrorParamSetSize(c_err, err, kernel.Name())) {
        return false;
    }

    // Launch the CUDA kernel
    const int nBlocksHor = _w / 16 + 1;
    const int nBlocksVer = _h / 16 + 1;
    cuFuncSetBlockShape(cudaKernel, 16, 16, 1);
    c_err = cuLaunchGrid(cudaKernel, nBlocksHor, nBlocksVer);
    if (_cudaErrorLaunchKernel(c_err, err, kernel.Name())) {
        return false;
    }
        
    return true;
}
//----------------------------------------------------------------------------//
std::string CUDAImpl::BoilerplateCode(Kernel::Ptr kernel) const 
{
    std::stringstream ss;

    // Indent string (used to indent arguments)
    ss << ",\n" << std::string(kernel->Name().size() + 1, ' ');
    const std::string indent = ss.str();
    ss.str("");

    // Header with arguments
    ss << "__global__ void\n" << kernel->Name() << "(";
    int argcount = 0;
    _BoilerplateBufferArgs(ss, kernel->InputBuffers(), indent, argcount, true);
    _BoilerplateBufferArgs(ss, kernel->OutputBuffers(),indent,argcount, false);
    _BoilerplateParamArgs(ss, kernel->ParamsInt(), "int", indent,argcount);
    _BoilerplateParamArgs(ss, kernel->ParamsFloat(), "float", indent,argcount);
    ss << indent << "const int width"
       << indent << "const int height)\n";

    // Code for index and dimension check
    ss << "{\n"
       << "    const int x = blockIdx.x * blockDim.x + threadIdx.x;\n"
       << "    const int y = blockIdx.y * blockDim.y + threadIdx.y;\n"
       << "\n"
       << "    // array index\n"
       << "    const int idx = x + width * y;\n"
       << "\n"
       << "    // inside image bounds check\n"
       << "    if (x >= width || y >= height) {\n"
       << "        return;\n"
       << "    }\n"
       << "\n";
   
    // Do half to float conversions  on input buffers (if needed)
    _BoilerplateHalfToFloat(ss, kernel->InputBuffers());

    // Starting kernel code, writing single value or vectors to all zero
    _BoilerplateKernelCode(ss, kernel->OutputBuffers());

    // Do float to half conversions on output buffers (if needed)
    _BoilerplateFloatToHalf(ss, kernel->OutputBuffers());
    
    ss << "}";
    return ss.str();
}
//----------------------------------------------------------------------------//
void CUDAImpl::_StartTimer()
{
    cudaEventRecord(_start, 0);
}
//----------------------------------------------------------------------------//
double CUDAImpl::_StopTimer()
{
    cudaEventRecord(_stop, 0);
    cudaEventSynchronize(_stop);
    float time;
    cudaEventElapsedTime(&time, _start, _stop);
    return time;
}
//----------------------------------------------------------------------------//
bool CUDAImpl::_FreeBuffers(std::string * err)
{
    cudaError_t c_err;
    std::map<std::string, float*>::iterator itb;
    for(itb = _cudaBuffers.begin(); itb != _cudaBuffers.end(); ++itb) {
        c_err = cudaFree(itb->second);
        if (_cudaErrorFree(c_err, err)) {
            return false;
        }
    }
    _cudaBuffers.clear();
    return true;
}
//----------------------------------------------------------------------------//
bool CUDAImpl::_UnloadModule(std::string * err)
{
    CUresult c_err;
    if (_cudaBuild) {
        _cudaKernels.clear();
        _cudaBuild = false;
        c_err = cuModuleUnload(_cudaModule);
        if (_cudaErrorLoadModule(c_err, err)) {
            return false;
        }
    }
    return true;
}
//----------------------------------------------------------------------------//
std::string _GetTypeStr(const Buffer::Ptr & buffer)
{
    std::stringstream type;
    switch(buffer->Type()) {
        case Buffer::UNSIGNED_BYTE:
            if (buffer->Channels() > 1) {
                type << "uchar";
            } else {
                type << "unsigned char";
            }
            break;
        case Buffer::HALF:
            type << "unsigned short";
            break;
        case Buffer::FLOAT:
        default:
            type << "float";
    };
    if (buffer->Channels() > 1 && buffer->Type() != Buffer::HALF) {
        type << buffer->Channels();
    }
    return type.str();
}
//----------------------------------------------------------------------------//
int _cudaGetMaxGflopsDeviceId()
{
	int device_count = 0;
	cudaGetDeviceCount( &device_count );

	cudaDeviceProp device_properties;
	int max_gflops_device = 0;
	int max_gflops = 0;
	
	int current_device = 0;
	cudaGetDeviceProperties( &device_properties, current_device );
	max_gflops = device_properties.multiProcessorCount *
            device_properties.clockRate;
	++current_device;

	while( current_device < device_count )
	{
		cudaGetDeviceProperties( &device_properties, current_device );
		int gflops = device_properties.multiProcessorCount *
                device_properties.clockRate;
		if( gflops > max_gflops )
		{
			max_gflops        = gflops;
			max_gflops_device = current_device;
		}
		++current_device;
	}

	return max_gflops_device;
}
//----------------------------------------------------------------------------//
int _execPipe(const char* cmd, std::string * err) {
#ifdef __unix__
    FILE* pipe = popen(cmd, "r");
#else
    FILE* pipe = _popen(cmd, "r");
#endif
    if (!pipe) {
        (*err) += "Could not execute command";
        (*err) += std::string(cmd);
        return 1;
    }
    char buffer[128];
    std::string result = "";
    while(!feof(pipe)) {
    	if(fgets(buffer, 128, pipe) != NULL)
    		result += buffer;
    }
#ifdef __unix__
    int exit_status = pclose(pipe);
#else
    int exit_status = _pclose(pipe);
#endif
    if (exit_status) {
        (*err) += result;
    }
    return exit_status;
}
//----------------------------------------------------------------------------//
int _removeFile(const char * filename)
{
    std::string command = std::string("rm ") + std::string(filename);
    return system(command.c_str());
}
//----------------------------------------------------------------------------//
CUresult _SetBufferArgs(const CUfunction & cudaKernel,
                        const std::vector<Kernel::BufferLink> & buffers,
                        std::map<std::string, float*> & cudaBuffers,
                        int & paramOffset)
{
    CUresult err = CUDA_SUCCESS;
    for(size_t i = 0; i < buffers.size(); ++i) {
        const Buffer::Ptr & targetBuffer = buffers[i].TargetBuffer();
        float * cudaBuffer = cudaBuffers.find(targetBuffer->Name())->second;
        err = cuParamSetv(cudaKernel,paramOffset, &cudaBuffer,sizeof(void*));
        paramOffset += sizeof(void *);
    }
    return err;
}
//----------------------------------------------------------------------------//
template<typename T>
CUresult _SetParamArgs(const CUfunction & cudaKernel,
                   const std::vector<Parameter<T> > & params,
                   int & paramOffset)
{
    CUresult err;
    for(size_t i = 0; i < params.size(); ++i) {
        T v = params[i].Value();
        err = cuParamSetv(cudaKernel,paramOffset, &v, sizeof(T));
        paramOffset += sizeof(T);
    }
    return err;
}
//----------------------------------------------------------------------------//
void _BoilerplateBufferArgs(
    std::stringstream & ss,
    const std::vector<Kernel::BufferLink> & bufferLinks,
    const std::string & indent,
    int & argcount,
    bool input)
{
    for(size_t i = 0; i < bufferLinks.size(); ++i) {
        //if first argument, don't indent
        if(!argcount++) {
            ss << indent;
        }

        // Only input buffers are declared const
        if(input) {
            ss << "const ";
        }

        const Buffer::Ptr & buffer = bufferLinks[i].TargetBuffer();
        
        // Actual pointer
        ss << _GetTypeStr(buffer)  << " * " << bufferLinks[i].Name();

        // Rename if half
        if(buffer->Type() == Buffer::HALF) {
            ss << "_half";
        }
    }
}
//----------------------------------------------------------------------------//
template<typename T>
void _BoilerplateParamArgs(std::stringstream & ss,
                           const std::vector<Parameter<T > > & params,
                           const char * typenameStr,
                           const std::string & indent,
                           int & argcount)
{
    for(size_t i = 0; i < params.size(); ++i) {
        if (!argcount++) {
            ss << indent;
        }
        ss << "const " << typenameStr << " " << params[i].Name();
    }
}
//----------------------------------------------------------------------------//
void _BoilerplateHalfToFloat(
    std::stringstream & ss,
    const std::vector<Kernel::BufferLink> & bufferLinks)
{
    int count = 0;
    for(size_t i = 0; i < bufferLinks.size(); ++i) {
        const Buffer::Ptr & buf = bufferLinks[i].TargetBuffer();
        const std::string & name = bufferLinks[i].Name();
        // if buffer is of type half, do conversion
        if (buf->Type() == Buffer::HALF) {
            if (!count++) { // only place comment once
                ss << "    // half to float conversion\n";
            }

            // Only 1 channel is a specialcase
            if (buf->Channels() == 1) {
                ss << "    const float " << name << " = "
                   << "__half2float(" << name << "_half[idx]);\n";
                continue;
            }

            // const floatX = make_floatX( 
            std::stringstream subss;
            subss << "    const float" << buf->Channels() << " "
                  << bufferLinks[i].Name()
                  << " = make_float" << buf->Channels() << "(";
            const std::string preindent = subss.str();
            ss << preindent;
            
            // Indent for every row in the make_floatX call
            subss.str("");
            subss << ",\n" << std::string(preindent.size(), ' ');
            const std::string subindent = subss.str();

            
            for (unsigned int j = 0; j < buf->Channels(); ++j) {
                if(!j) { // only subindent after first row
                    ss << subindent;
                }

                // half2float conversion from single value unsigned short
                ss << "__half2float(" <<  name << "_half["
                   << buf->Channels() << " * idx + " << j << "])";
            }
            ss <<");\n";
        }
    }

    // If any half to float conversions were added, end with a new line
    if (count) {
        ss << "\n";
    }
}
//----------------------------------------------------------------------------//
void _BoilerplateKernelCode(
    std::stringstream & ss,
    const std::vector<Kernel::BufferLink> & bufferLinks)
{
    ss << "    // kernel code\n";
    for (size_t i = 0; i < bufferLinks.size(); ++i) {
        Buffer::Ptr b = bufferLinks[i].TargetBuffer();
        ss << "    ";

        // If half, we need to declare a new variable
        if (b->Type() == Buffer::HALF) {
            ss << "float";

            // If more than 1 channel, then it's a vector
            if(b->Channels() > 1) {
                ss << b->Channels();
            }
            ss << " ";
        }

        // variale name
        ss << bufferLinks[i].Name();

        // if not half, then we write directly to the global memory
        if (b->Type() != Buffer::HALF) {
            ss << "[idx]";
        }
        
        ss << " = ";
        if (b->Channels() == 1) { // single value -> just a zero
            ss << "0;\n";
        } else { // if not single value -> vector
            ss << "make_";
            if (b->Type() != Buffer::HALF) {
                ss << _GetTypeStr(b);
            } else {
                ss << "float" << b->Channels();
            }

            // populate vector with zeros
            ss << "(";
            for (size_t j = 0; j < b->Channels(); ++j) {
                ss << (j ==0 ? "" : ", ") << "0";
            }
            ss << ");\n";
        }
    }
}
//----------------------------------------------------------------------------//
void _BoilerplateFloatToHalf(
    std::stringstream & ss,
    const std::vector<Kernel::BufferLink> & bufferLinks)
{
    // Do half to float conversions (if needed)
    int count = 0;
    for(size_t i = 0; i < bufferLinks.size(); ++i) {
        const Buffer::Ptr & buf = bufferLinks[i].TargetBuffer();
        const std::string name = bufferLinks[i].Name();

        // if buffer pixel type is half, do conversion
        if (buf->Type() == Buffer::HALF) {
            //only write count once
            if (!count++) {
                    ss << "\n    // float to half conversion\n";
            }

            // need to write every channel individually
            for (unsigned int j = 0; j < buf->Channels(); ++j) {
                // variable name (defined in arguments
                ss << "    " << name << "_half";

                // index
                ss << "[";
                if (buf->Channels() > 1) {
                    ss << buf->Channels() << " * ";
                }
                ss << "idx + " << j << "]";

                // conversion from float4
                ss << "= __float2half_rn(" << name;
                switch(j) {
                    case 0:
                        ss << (buf->Channels() > 1 ? ".x" : "");
                        break;
                    case 1:
                        ss << ".y";
                        break;
                    case 2:
                        ss << ".z";
                        break;
                    case 3:
                        ss << ".w";
                        break;
                }
                ss << ");\n";
            }
        }
    }
}
//----------------------------------------------------------------------------//
} // end namespace gpuip
//----------------------------------------------------------------------------//
