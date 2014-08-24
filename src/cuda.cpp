#include "cuda.h"
#include "cuda_error.h"
#include "helper_math.cuh"
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
inline int _cudaGetMaxGflopsDeviceId();
//----------------------------------------------------------------------------//
inline std::string _GetTypeStr(Buffer::Ptr buffer);
//----------------------------------------------------------------------------//
ImageProcessor * CreateCUDA()
{
    return new CUDAImpl();
}
//----------------------------------------------------------------------------//
CUDAImpl::CUDAImpl()
        : ImageProcessor(CUDA), _cudaBuild(false)
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
        _cudaBuffers[it->second->name] = NULL;
        cudaError_t c_err = cudaMalloc(&_cudaBuffers[it->second->name],
                                       _BufferSize(it->second));
        if(_cudaErrorMalloc(c_err, err)) {
            return GPUIP_ERROR;
        }
    }
    return _StopTimer();
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
inline void _removeFile(const char * filename)
{
    std::string command = std::string("rm ") + std::string(filename);
    system(command.c_str());
}
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
#ifdef CUDA_HELPER_DIR
    out << "#include <" << file_helper_math_h << ">\n";
#endif
    out << "extern \"C\" { \n"; // To avoid function name mangling 
    for(size_t i = 0; i < _kernels.size(); ++i) {
        out << _kernels[i]->code << "\n";
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
       << " --Wno-deprecated-gpu-targets";
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
    std::string pipe_err;
    int nvcc_exit_status = _execPipe(ss.str().c_str(), &pipe_err);

    // Cleanup temp text file
    _removeFile(file_helper_math_h);
    _removeFile(file_temp_cu);
    
    if (nvcc_exit_status) {
        (*err) = "Cuda error: Could not compile kernels.\n";
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
                                    _kernels[i]->name.c_str());
        if (_cudaErrorGetFunction(c_err, err, _kernels[i]->name)) {
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
        e =cudaMemcpy(data, _cudaBuffers[buffer->name],
                      size, cudaMemcpyDeviceToHost);
    } else if (op == Buffer::COPY_TO_GPU) {
        e = cudaMemcpy(_cudaBuffers[buffer->name],data,
                       size, cudaMemcpyHostToDevice);
    }
    if (_cudaErrorCopy(e, err, buffer->name, op)) {
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
    for(size_t i = 0; i < kernel.inBuffers.size(); ++i) {
        c_err = cuParamSetv(cudaKernel, paramOffset,
                            &_cudaBuffers[kernel.inBuffers[i].buffer->name],
                            sizeof(void*));
        paramOffset += sizeof(void *);
    }
    for(size_t i = 0; i < kernel.outBuffers.size(); ++i) {
        c_err = cuParamSetv(cudaKernel, paramOffset,
                            &_cudaBuffers[kernel.outBuffers[i].buffer->name],
                            sizeof(void*));
        paramOffset += sizeof(void *);
    }
    for(size_t i = 0; i < kernel.paramsInt.size(); ++i) {
        c_err = cuParamSetv(cudaKernel, paramOffset,
                            &kernel.paramsInt[i].value, sizeof(int));
        paramOffset += sizeof(int);
    }
    for(size_t i = 0; i < kernel.paramsFloat.size(); ++i) {
        c_err = cuParamSetv(cudaKernel, paramOffset,
                            &kernel.paramsFloat[i].value, sizeof(float));
        paramOffset += sizeof(float);
    }
    // int and width parameters
    c_err = cuParamSetv(cudaKernel, paramOffset, &_w, sizeof(int));
    paramOffset += sizeof(int);
    c_err = cuParamSetv(cudaKernel, paramOffset, &_h, sizeof(int));
    paramOffset += sizeof(int);
    
    // It should be fine to check once all the arguments have been set
    if(_cudaErrorCheckParamSet(c_err, err, kernel.name)) {
        return false;
    }
    
    c_err = cuParamSetSize(cudaKernel, paramOffset);
    if (_cudaErrorParamSetSize(c_err, err, kernel.name)) {
        return false;
    }

    // Launch the CUDA kernel
    const int nBlocksHor = _w / 16 + 1;
    const int nBlocksVer = _h / 16 + 1;
    cuFuncSetBlockShape(cudaKernel, 16, 16, 1);
    c_err = cuLaunchGrid(cudaKernel, nBlocksHor, nBlocksVer);
    if (_cudaErrorLaunchKernel(c_err, err, kernel.name)) {
        return false;
    }
        
    return true;
}
//----------------------------------------------------------------------------//
std::string CUDAImpl::BoilerplateCode(Kernel::Ptr kernel) const 
{
    std::stringstream ss;

    // Indent string
    ss << ",\n" << std::string(kernel->name.size() + 1, ' ');
    const std::string indent = ss.str();
    ss.str("");
   
    ss << "__global__ void\n" << kernel->name << "(";

    bool first = true;

    for(size_t i = 0; i < kernel->inBuffers.size(); ++i) {
        ss << (first ? "" : indent);
        first = false;
        const std::string & name = kernel->inBuffers[i].buffer->name;
        ss << "const " << _GetTypeStr(_buffers.find(name)->second)
           << " * " << kernel->inBuffers[i].name
           << (_buffers.find(name)->second->type == Buffer::HALF ? "_half" : "");   }

    for(size_t i = 0; i < kernel->outBuffers.size(); ++i) {
        ss << (first ? "" : indent);
        first = false;
        const std::string & name = kernel->outBuffers[i].buffer->name;
        ss <<  _GetTypeStr(_buffers.find(name)->second)
           << " * " << kernel->outBuffers[i].name
           << (_buffers.find(name)->second->type == Buffer::HALF ? "_half" : "");
    }

    for(size_t i = 0; i < kernel->paramsInt.size(); ++i) {
        ss << (first ? "" : indent);
        first = false;        
        ss << "const int " << kernel->paramsInt[i].name;
    }
    for(size_t i = 0; i < kernel->paramsFloat.size(); ++i) {
        ss << (first ? "" : indent);
        first = false;
        ss << "const float " << kernel->paramsFloat[i].name;
    }
    ss << indent << "const int width" << indent << "const int height)\n";
  
    ss << "{\n";
    ss << "    const int x = blockIdx.x * blockDim.x + threadIdx.x;\n";
    ss << "    const int y = blockIdx.y * blockDim.y + threadIdx.y;\n\n";
    ss << "    // array index\n";
    ss << "    const int idx = x + width * y;\n\n";
    ss << "    // inside image bounds check\n";
    ss << "    if (x >= width || y >= height) {\n";
    ss << "        return;\n";
    ss << "    }\n\n";
   
    // Do half to float conversions (if needed)
    for(size_t i = 0; i < kernel->inBuffers.size(); ++i) {
        const std::string & bname = kernel->inBuffers[i].buffer->name;
        Buffer::Ptr buf = _buffers.find(bname)->second;
        if (buf->type == Buffer::HALF) {
            if (!i) {
                ss << "    // half to float conversion\n";
            }
            
            if (buf->channels == 1) {
                ss << "    const float " << kernel->inBuffers[i].name
                   << " = " << "__half2float(" <<  kernel->inBuffers[i].name
                   << "_half[idx]);\n";
                continue;
            }
            
            std::stringstream subss;
            subss << "    const float" << buf->channels << " "
                  << kernel->inBuffers[i].name
                  << " = make_float" << buf->channels << "(";
            const std::string preindent = subss.str();
            subss.str("");
            subss << ",\n" << std::string(preindent.size(), ' ');
            const std::string subindent = subss.str();
            ss << preindent;
            bool subfirst = true;
            for (unsigned int j = 0; j < buf->channels; ++j) {
                ss << (subfirst ? "" : subindent);
                subfirst = false;
                ss << "__half2float(" <<  kernel->inBuffers[i].name
                   << "_half[" << buf->channels << " * idx + " << j << "])";
            }
            ss <<");\n";

            if (i == kernel->inBuffers.size() - 1) {
                ss << "\n";
            }
        }
    }

    ss << "    // kernel code\n";
    for (size_t i = 0; i < kernel->outBuffers.size(); ++i) {
        Buffer::Ptr b = _buffers.find(
            kernel->outBuffers[i].buffer->name)->second;
        ss << "    ";
        if (b->type == Buffer::HALF) {
            ss << "float" << b->channels << " ";
        }
        ss << kernel->outBuffers[i].name;
        if (b->type != Buffer::HALF) {
            ss << "[idx]";
        }
        ss << " = ";
        if (b->channels == 1) {
            ss << "0;\n";
        } else {
            ss << "make_";
            if (b->type != Buffer::HALF) {
                ss << _GetTypeStr(b);
            } else {
                ss << "float" << b->channels;
            }
            ss << "(";
            for (size_t j = 0; j < b->channels; ++j) {
                ss << (j ==0 ? "" : ", ") << "0";
            }
            ss << ");\n";
        }
    }

    // Do half to float conversions (if needed)
    for(size_t i = 0; i < kernel->outBuffers.size(); ++i) {
        const std::string & bname = kernel->outBuffers[i].buffer->name;
        Buffer::Ptr buf = _buffers.find(bname)->second;
        if (buf->type == Buffer::HALF) {
            if (!i) {
                ss << "\n    // float to half conversion\n";
            }

            for (unsigned int j = 0; j < buf->channels; ++j) {
                ss << "    " << kernel->outBuffers[i].name << "_half[";
                if (buf->channels > 1) {
                    ss << buf->channels << " * ";
                }
                ss << "idx + " << j << "] = __float2half_rn("
                   << kernel->outBuffers[i].name;
                switch(j) {
                    case 0:
                        ss << (buf->channels > 1 ? ".x" : "");
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
std::string _GetTypeStr(Buffer::Ptr buffer)
{
    std::stringstream type;
    switch(buffer->type) {
        case Buffer::UNSIGNED_BYTE:
            if (buffer->channels > 1) {
                type << "uchar";
            } else {
                type << "unsigned char";
            }
            break;
        case Buffer::HALF:
            type << "unsigned short";
            break;
        case Buffer::FLOAT:
            type << "float";
            break;
        default:
            type << "float";
    };
    if (buffer->channels > 1 && buffer->type != Buffer::HALF) {
        type << buffer->channels;
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
} // end namespace gpuip
//----------------------------------------------------------------------------//
