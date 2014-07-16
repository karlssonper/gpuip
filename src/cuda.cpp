#include "cuda.h"
#include "cuda_error.h"
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
inline int _cudaGetMaxGflopsDeviceId();
//----------------------------------------------------------------------------//
Base * CreateCUDA()
{
    return new CUDAImpl();
}
//----------------------------------------------------------------------------//
CUDAImpl::CUDAImpl()
  : Base(CUDA)
{
    if (cudaSetDevice(_cudaGetMaxGflopsDeviceId()) != cudaSuccess) {
        throw std::logic_error("gpuip::CUDAImpl() could not set device id");
    };
}
//----------------------------------------------------------------------------//
bool CUDAImpl::InitBuffers(std::string * err)
{
    cudaError_t c_err;
    std::map<std::string,Buffer>::const_iterator it;
    for (it = _buffers.begin(); it != _buffers.end(); ++it) {
        _cudaBuffers[it->second.name] = NULL;
        c_err = cudaMalloc(&_cudaBuffers[it->second.name],
                           _GetBufferSize(it->second));
        if(_cudaErrorMalloc(c_err, err)) {
            return false;
        }
    }
    return true;
}
//----------------------------------------------------------------------------//
bool CUDAImpl::Build(std::string * err)
{
    // Create temporary file to compile
    std::ofstream out(".temp.cu");
    out << "extern \"C\" { \n"; // To avoid function name mangling 
    for (int i = 0; i < _kernels.size(); ++i) {
        out << _kernels[i]->code << "\n";
    }
    out << "}"; // End the extern C bracket
    out.close();

    int nvcc_exit_status = system(
        "nvcc -ptx .temp.cu -o .temp.ptx --Wno-deprecated-gpu-targets");

    // Cleanup temp text file
    system("rm .temp.cu");
    
    if (nvcc_exit_status) {
        (*err) = "Cuda error: Could not compile kernels.";
        return false;
    }

    // Load cuda ptx from file
    CUmodule module;
    CUresult c_err = cuModuleLoad(&module, ".temp.ptx");
    system("rm .temp.ptx");
    if (_cudaErrorLoadModule(c_err, err)) {
        return false;
    }

    _cudaKernels.resize(_kernels.size());
    for (int i = 0; i < _kernels.size(); ++i) {
        c_err = cuModuleGetFunction(&_cudaKernels[i], module,
                                    _kernels[i]->name.c_str());
        if (_cudaErrorGetFunction(c_err, err, _kernels[i]->name)) {
            return false;
        }
    }
    
    return true;
}
//----------------------------------------------------------------------------//
bool CUDAImpl::Process(std::string * err)
{
    for(int i = 0; i < _kernels.size(); ++i) {
        if (!_LaunchKernel(*_kernels[i].get(), _cudaKernels[i], err)) {
            return false;
        }
    }
    return true;
}
//----------------------------------------------------------------------------//
bool CUDAImpl::Copy(const std::string & buffer,
                    Buffer::CopyOperation op,
                    void * data,
                    std::string * err)
{
    cudaError_t e;
    const size_t size = _GetBufferSize(_buffers[buffer]);
    if (op == Buffer::READ_DATA) {
        e =cudaMemcpy(data, _cudaBuffers[buffer], size, cudaMemcpyDeviceToHost);
    } else if (op == Buffer::WRITE_DATA) {
        e = cudaMemcpy(_cudaBuffers[buffer],data, size, cudaMemcpyHostToDevice);
    }
    if (_cudaErrorCopy(e, err, buffer, op)) {
        return false;
    }
    return true;
}
//----------------------------------------------------------------------------//
std::string CUDAImpl::GetBoilerplateCode(Kernel::Ptr kernel) const 
{
   std::stringstream ss;

   ss << "__global__ void\n" << kernel->name << "(";

   // whitespace before each param (indent)
   const int ws = kernel->name.size() + 1;

   bool first = true;
   _GetBoilerplateCodeBuffers(ss, kernel->inBuffers, true, first, ws);
   _GetBoilerplateCodeBuffers(ss, kernel->outBuffers, false, first, ws);

   for (int i = 0; i < kernel->paramsInt.size(); ++i) {
       if (first) {
           first = false;
       } else {
           ss << ",\n" << std::string(ws, ' ');
       }
        
       ss << "int " << kernel->paramsInt[i].name;
   }
   for (int i = 0; i < kernel->paramsFloat.size(); ++i) {
       if (first) {
           first = false;
       } else {
           ss << ",\n" << std::string(ws, ' ');;
       }
       ss <<  "float " << kernel->paramsFloat[i].name;
   }
    
   ss << ")\n{\n}";
    
   return ss.str();
}
//----------------------------------------------------------------------------//
void CUDAImpl::_GetBoilerplateCodeBuffers(
    std::stringstream & ss,
    const std::vector<std::pair<Buffer, std::string> > & buffers,
    const bool inBuffer,
    bool & first,
    const int indent) const
{
    for(int i = 0; i < buffers.size(); ++i) {
        if (first) {
            first = false;
        } else {
            ss << ",\n"  << std::string(indent, ' ');
        }
        const Buffer & buffer = _buffers.find(buffers[i].first.name)->second;
        std::string type;
        switch(buffer.bpp/buffer.channels) {
            case 1:
                type = "unsigned char";
                break;
            case 4:
                type = "float";
                break;
            case 8:
                type = "double";
                break;
            default:
                type = "float";
        };
    
        ss << (inBuffer ? "const " : "") << type
           << " * " << buffers[i].second;
    }
}
//----------------------------------------------------------------------------//
bool CUDAImpl::_LaunchKernel(Kernel & kernel,
                             const CUfunction & cudaKernel,
                             std::string * err)
{
    // Set CUDA kernel arguments
    CUresult c_err;
    int paramOffset = 0;
    for (size_t i = 0; i < kernel.inBuffers.size(); ++i) {
        c_err = cuParamSetv(cudaKernel, paramOffset,
                            &_cudaBuffers[kernel.inBuffers[i].first.name],
                            sizeof(void*));
        paramOffset += sizeof(void *);
    }
    for (size_t i = 0; i < kernel.outBuffers.size(); ++i) {
        c_err = cuParamSetv(cudaKernel, paramOffset,
                            &_cudaBuffers[kernel.outBuffers[i].first.name],
                            sizeof(void*));
        paramOffset += sizeof(void *);
    }
    for(int i = 0; i < kernel.paramsInt.size(); ++i) {
        c_err = cuParamSetv(cudaKernel, paramOffset,
                            &kernel.paramsInt[i].value, sizeof(int));
        paramOffset += sizeof(int);
    }
    for(int i = 0; i < kernel.paramsFloat.size(); ++i) {
        c_err = cuParamSetv(cudaKernel, paramOffset,
                            &kernel.paramsFloat[i].value, sizeof(float));
        paramOffset += sizeof(float);
    }

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
