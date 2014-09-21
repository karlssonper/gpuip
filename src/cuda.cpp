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
namespace gpuip {
//----------------------------------------------------------------------------//
inline int _cudaGetMaxGflopsDeviceId();
//----------------------------------------------------------------------------//
inline std::string _GetTypeStr(Buffer::Ptr buffer);
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
    if(!_ValidateBuffers(err)) {
        return GPUIP_ERROR;
    }
    
    _StartTimer();

    if (!_FreeBuffers(err)) {
        return GPUIP_ERROR;
    }

    CUresult result;
    std::map<std::string,Buffer::Ptr>::const_iterator it;
    for(it = _buffers.begin(); it != _buffers.end(); ++it) {
        CUdeviceptr dptr;
        
        // If tagged as texture, get texture handle
        if(it->second->isTexture) {
            if (!_cudaBuild) {
                (*err) += "CUDA error: Need to build kernel code ";
                (*err) += "before allocating.\n";
                return GPUIP_ERROR;
            }

            size_t pitch;
            result = cuMemAllocPitch(&dptr, &pitch,
                                     it->second->width * 4,
                                     it->second->height, 4);
            _cudaPitch[it->second->name] = pitch;
            if(_cudaErrorMemAlloc(result, err, it->second->name)) {
                return GPUIP_ERROR;
            }
            
            CUtexref texRef;
            result = cuModuleGetTexRef(&texRef, _cudaModule,
                                                it->second->name.c_str());
            if(_cudaErrorGetTexRef(result, err,it->second->name)) {
                return GPUIP_ERROR;
            }

            CUDA_ARRAY_DESCRIPTOR desc;
            desc.Format = CU_AD_FORMAT_FLOAT;
            desc.Width = it->second->width;
            desc.Height = it->second->height;
            desc.NumChannels = it->second->channels;
            CUresult res;
            res = cuTexRefSetAddress2D(texRef, &desc, dptr, pitch);
            if(res != CUDA_SUCCESS) {
                std::cout << CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT << std::endl;
                std::cout << CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT << std::endl;
                std::cout << "NOOOO" << res <<  std::endl;
            };
            //todo not * 4
            
            cuTexRefSetFlags(texRef, CU_TRSF_NORMALIZED_COORDINATES);
            cuTexRefSetAddressMode(texRef, 0, CU_TR_ADDRESS_MODE_CLAMP);
            cuTexRefSetAddressMode(texRef, 1, CU_TR_ADDRESS_MODE_CLAMP);
            cuTexRefSetFilterMode(texRef, CU_TR_FILTER_MODE_LINEAR);
                        
            _cudaTextures[it->second->name] = texRef;
        } else {
            result = cuMemAlloc(&dptr, _BufferSize(it->second));
            if(_cudaErrorMemAlloc(result, err, it->second->name)) {
                return GPUIP_ERROR;
            }
        }
        _cudaBuffers[it->second->name] = dptr;
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
inline int _removeFile(const char * filename)
{
    std::string command = std::string("rm ") + std::string(filename);
    return system(command.c_str());
}
double CUDAImpl::Build(std::string * err)
{
    if(!_ValidateKernels(err)) {
        return GPUIP_ERROR;
    }
    
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

    // Buffers tagged as textures
    std::map<std::string,Buffer::Ptr>::const_iterator it;
    for(it = _buffers.begin(); it != _buffers.end(); ++it) {
        if(it->second->isTexture) {
            out << "texture<";
            out << "float";
            out << ",2> " << it->second->name << ";\n";
        }
    }
    
    // Kernels
    for(size_t i = 0; i < _kernels.size(); ++i) {
        //Defines for input buffers taggad as textures
        std::vector<std::string> definedTextures;
        for(size_t j = 0; j < _kernels[i]->inBuffers.size(); ++j) {
            if (_kernels[i]->inBuffers[j].buffer->isTexture) {
                definedTextures.push_back(_kernels[i]->inBuffers[j].name);
                out << "#define " << definedTextures.back()
                    << " " << _kernels[i]->inBuffers[j].buffer->name << "\n";
            }
        }
        
        out << _kernels[i]->code << "\n";

        //Undef buffers tagged as textures
        for(size_t j = 0; j < definedTextures.size(); ++j) {
            out << "#undef " << definedTextures[j] << "\n";
        }
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
    //_removeFile(file_temp_cu);
        
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
    CUresult c_err = CUDA_SUCCESS;
    const size_t size = _BufferSize(buffer);
    if (op == Buffer::COPY_FROM_GPU) {
        c_err = cuMemcpyDtoH(data, _cudaBuffers[buffer->name], size);
    } else if (op == Buffer::COPY_TO_GPU) {
        if(buffer->isTexture) {
            const CUDA_MEMCPY2D cpy = {
                0,
                0,
                CU_MEMORYTYPE_HOST,
                data,
                0,
                0,
                0,

                0,
                0,
                CU_MEMORYTYPE_DEVICE,
                0,
                _cudaBuffers[buffer->name],
                0,
                _cudaPitch[buffer->name],
                buffer->width*4,
                buffer->height};
            cuMemcpy2D(&cpy);
        } else {
            c_err = cuMemcpyHtoD(_cudaBuffers[buffer->name], data, size);
        }
    }
    if (_cudaErrorCopy(c_err, err, buffer->name, op)) {
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
        if (kernel.inBuffers[i].buffer->isTexture) {
            // Make texture available in kernel
            c_err = cuParamSetTexRef(
                cudaKernel, CU_PARAM_TR_DEFAULT,
                _cudaTextures[kernel.inBuffers[i].buffer->name]);
        } else {            
            c_err = cuParamSetv(cudaKernel, paramOffset,
                                &_cudaBuffers[kernel.inBuffers[i].buffer->name],
                                sizeof(CUdeviceptr));
            paramOffset += sizeof(CUdeviceptr);
        }
    }
    for(size_t i = 0; i < kernel.outBuffers.size(); ++i) {
        c_err = cuParamSetv(cudaKernel, paramOffset,
                            &_cudaBuffers[kernel.outBuffers[i].buffer->name],
                            sizeof(CUdeviceptr));
        paramOffset += sizeof(CUdeviceptr);
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
    int width = kernel.outBuffers.front().buffer->width;
    int height = kernel.outBuffers.front().buffer->height;
    c_err = cuParamSetv(cudaKernel, paramOffset, &width, sizeof(int));
    paramOffset += sizeof(int);
    c_err = cuParamSetv(cudaKernel, paramOffset, &height, sizeof(int));
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
    const int nBlocksHor = width / 16 + 1;
    const int nBlocksVer = width / 16 + 1;
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

    // Textures
    for(size_t i = 0; i < kernel->inBuffers.size(); ++i) {
        if(kernel->inBuffers[i].buffer->isTexture) {
            ss << "//texture<float,2> " << kernel->inBuffers[i].name
               << "; defined globally, do not uncomment.\n";
        }
    }
    
    ss << "__global__ void\n" << kernel->name << "(";

    bool first = true;

    for(size_t i = 0; i < kernel->inBuffers.size(); ++i) {
        if(!kernel->inBuffers[i].buffer->isTexture) {
            ss << (first ? "" : indent);
            first = false;
            const std::string & name = kernel->inBuffers[i].buffer->name;
            Buffer::Type type = _buffers.find(name)->second->type;
            ss << "const " << _GetTypeStr(_buffers.find(name)->second)
               << " * " << kernel->inBuffers[i].name
               << (type == Buffer::HALF ? "_half" : "");
        }
    }

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
    CUresult c_err;
    std::map<std::string, CUdeviceptr>::iterator itb;
    for(itb = _cudaBuffers.begin(); itb != _cudaBuffers.end(); ++itb) {
        c_err = cuMemFree(itb->second);
        if (_cudaErrorMemFree(c_err, err, itb->first)) {
            return false;
        }
    }
    _cudaBuffers.clear();
    _cudaTextures.clear(); // only handles, no need to free/delete memory
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
