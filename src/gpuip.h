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

#ifndef GPUIP_H_
#define GPUIP_H_
//----------------------------------------------------------------------------//
#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <stdexcept>
//----------------------------------------------------------------------------//
#ifdef _GPUIP_PYTHON_BINDINGS
#include <boost/shared_ptr.hpp>
#else
#include <tr1/memory>
#endif
//----------------------------------------------------------------------------//
/*! \file */
/*! gpuip is a cross-platform framework for image processing on the GPU. */
namespace gpuip {
//----------------------------------------------------------------------------//
/*! If a function that is supposed to return execution time has failed,
  it will return this value instead. */
#define GPUIP_ERROR -1.0
//----------------------------------------------------------------------------//
/*! Different GPU environments available. */
enum GpuEnvironment {
    /*! <a href="https://www.khronos.org/opencl/">
      OpenCL, Open Computing Language by Khronos Group.</a>  */
    OpenCL,

    /*! <a href="https://developer.nvidia.com/cuda-zone">
      CUDA, Compute Unified Device Architecture by NVIDIA Corporation.</a>*/
    CUDA,

    /*! <a href="http://www.opengl.org/">
      GLSL, OpenGL Shading Language by Khronos Group.</a> */
    GLSL };
//----------------------------------------------------------------------------//
/*!
  \struct Buffer
  \brief A chunk of memory allocated on the GPU.

  This class 
 */
struct Buffer
{
    /*! \brief Smart pointer. */
#ifdef _GPUIP_PYTHON_BINDINGS
    typedef boost::shared_ptr<Buffer> Ptr;
#else
    typedef std::tr1::shared_ptr<Buffer> Ptr;
#endif
    /*! Operation when copying memory between CPU and GPU */
    enum CopyOperation{
        /*! Copy data from CPU to GPU */
        COPY_TO_GPU,
        /*! Copy data from GPU to CPU */
        COPY_FROM_GPU };

    /*! \brief Supported data types */
    enum Type{
        /*! 8 bits per channel. Used in png, jpeg, tga, tiff formats */
        UNSIGNED_BYTE,
        
        /*! 16 bits per channel. Used in exr images. */
        HALF,

        /*! 32 bits per channel. Used in exr images, typically when half is
          not supported in the current environemnt. */
        FLOAT };
    
    Buffer(const std::string & name, Type type, unsigned int channels);

    /*! \brief Unique identifying name.

      Each buffer has a unique name. The buffer can still be called referenced
      as something else in a kernel. */
    const std::string name;
    
    Type type;

    /*! \brief Channels of data per pixel.

      A typical RGBA image has 4 channels. Gpuip buffers  with 2 or 3 channels
      have not been tested as much as 1 or 4 channel buffers. */
    unsigned int channels;
};
//----------------------------------------------------------------------------//
/*!
  \struct Parameter
  \brief A parameter has a name and a value.
*/
template<typename T>
struct Parameter
{
    Parameter(const std::string & n, T v) : name(n), value(v) {}
    std::string name;
    T value;
};
//----------------------------------------------------------------------------//
/*!
  \struct Kernel
  \brief 
*/
struct Kernel
{
    /*! \brief Smart pointer. */
#ifdef _GPUIP_PYTHON_BINDINGS
    typedef boost::shared_ptr<Kernel> Ptr;
#else
    typedef std::tr1::shared_ptr<Kernel> Ptr;
#endif
    /*!
      \struct BufferLink
      \brief Tells a kernel which buffers to use in the argument list
    */
    struct BufferLink
    {
        BufferLink(Buffer::Ptr buffer_, const std::string & name_);
        
        /*! \brief Buffer to be used in the kernel arguments list. */
        Buffer::Ptr buffer;

        /*! \brief The name of buffer in kernel arguments list.
          
          This does not have to be the same as the Buffer::name.*/
        std::string name;
    };
    Kernel(const std::string & name);

    /*! \brief Unique identifying name.

      Each kernel must have a unique name.
      The kernel function in OpenCL and CUDA has to have the same name.
     */
    const std::string name;

    /*! \brief Kernel code.

     Must be set before the ImageProcessor::Build call.*/
    std::string code;

    /*! \brief Buffers used for input data. Can not be modified.

     Must be set before the ImageProcessor::Run call.*/
    std::vector<BufferLink> inBuffers;

    /*! \brief Buffers used to output data. Can not be read from.

     Must be set before the ImageProcessor::Run call. */
    std::vector<BufferLink> outBuffers;

    /*! \brief Integer parameters.

     Must be set before the ImageProcessor::Run call. */
    std::vector<Parameter<int> > paramsInt;

    /*! \brief Float parameters.

     Must be set before the ImageProcessor::Run call. */
    std::vector<Parameter<float> > paramsFloat;
};
//----------------------------------------------------------------------------//
/*!
  \class ImageProcessor
  \brief
*/
class ImageProcessor
{
  public:
    /*! \brief Smart pointer. */
#ifdef _GPUIP_PYTHON_BINDINGS
    typedef boost::shared_ptr<ImageProcessor> Ptr;
#else
    typedef std::tr1::shared_ptr<ImageProcessor> Ptr;
#endif

    /*! \brief Factory function to create an ImageProcessor entity. */
    static ImageProcessor::Ptr Create(GpuEnvironment env);
    
    virtual ~ImageProcessor() {}

    /*! \brief Check if gpuip was compiled with a GpuEnvironment. */
    static bool CanCreate(GpuEnvironment env);

    /*! \brief Returns the current GpuEnvironment. */
    GpuEnvironment Environment() const
    {
        return _env;
    }

    /*! \brief Set the dimensions of algorithms. Must be set explicitly. */
    void SetDimensions(unsigned int width, unsigned int height);

    /*! \brief Returns the images width in number of pixels */
    unsigned int Width() const
    {
        return _w;
    }

    /*! \brief Returns the images height in number of pixels */
    unsigned int Height() const
    {
        return _h;
    }

    /*! \brief Creates a Buffer object with allocation info

      \param name Unique identifying name of buffer
      \param type per channel data type
      \param channels number of channels of data per pixel

      \return A smart pointer to new registered Buffer object
      
      This is the only way a buffer can be registered to an ImageProcessor.
      Modifications can be made to the Buffer object as long as they are made
      before the ImageProcessor::Allocate call.      
     */
    Buffer::Ptr CreateBuffer(const std::string & name,
                             Buffer::Type type,
                             unsigned int channels);

    /*! \brief Creates and registeres a Kernel object.

      \param name Unique identifying name of the kernel

      \return A smart pointer to new registered Kernel object

      This is the only way a kernel can be registered to an ImageProcessor.
      Kernels will be run in the order they are created.
      Modifications can be made to the Kernel object as long as Kernel::code
      is set before the ImageProcessor::Build call and Kernel::inBuffers,
      Kernel::outBuffers, Kernel::paramsInt, Kernel::paramsFloat are set
      before the ImageProcessor::Run call.
     */
    Kernel::Ptr CreateKernel(const std::string & name);

    /*! \brief Allocates needed memory on the GPU.
      \param error if function fails, the explaining error string is stored here
      \return execution time in milliseconds. \ref GPUIP_ERROR on failure

      Call this function once all buffers have been created. Once things have
      been allocated on the GPU, ImageProcessor::Copy can be called. This
      function can be called multiple times since it starts
      with resetting previous allocated memory.
    */
    virtual double Allocate(std::string * error);

    /*! \brief Compiles the Kernel::Code for each Kernel object.
      \param error if function fails, the explaining error string is stored here
      \return execution time in milliseconds. \ref GPUIP_ERROR on failure
      
      Call this function once the Kernel::code has been set for all kernels.
      Can be called multiple times to rebuild the kernels.
    */
    virtual double Build(std::string * error);

    /*! \brief Runs all of the image processing kernels.
      \param error if function fails, the explaining error string is stored here
      \return execution time in milliseconds. \ref GPUIP_ERROR on failure

      Runs all the image processing kernels in the order they were created.
      ImageProcessor::Build and ImageProcessor::Allocate must have called
      before this function.
    */
    virtual double Run(std::string * error);

    /*! \brief Data tranfser from the CPU and the GPU.
      \param buffer buffer on the gpu to copy to/from
      \param operation decides if the copy is from the gpu or to the gpu
      \param data points to allocated memory on the CPU
      \param error if function fails, the explaining error string is stored here
      \return execution time in milliseconds. \ref GPUIP_ERROR on failure

      Copies data between the CPU and the GPU. User must guarantee that \c data
      is pointing to a memory space on the CPU with enough allocated memory.
      ImageProcessor::Allocate must be called at least once before this
      function.
    */
    virtual double Copy(Buffer::Ptr buffer,
                        Buffer::CopyOperation operation,
                        void * data,
                        std::string * error);

    /*! \brief Returns a boilerplate code for a given kernel.
      \param kernel Kernel to be processed
      \return boilerplate code

      Processes the kernel and the buffers to produce boilerplate code.
      Boilerplate code is often a good starting step when writing gpu kernels
      since they remove some of the redundent code that is shared between
      all kernels. It also guarantees that the argument list is correct.
     */
    virtual std::string BoilerplateCode(Kernel::Ptr kernel) const;
               
  protected:
    ImageProcessor(GpuEnvironment env);

    const GpuEnvironment _env;
    unsigned int _w; // width
    unsigned int _h; // height
    std::map<std::string, Buffer::Ptr> _buffers;
    std::vector<Kernel::Ptr> _kernels;

    unsigned int _BufferSize(Buffer::Ptr buffer) const;
  
  private:
    ImageProcessor();
    ImageProcessor(const ImageProcessor &);
    void operator=(const ImageProcessor &);
};
//----------------------------------------------------------------------------//
} //end namespace gpuip
//----------------------------------------------------------------------------//
#endif
