#include "io_wrapper.h"
#include "gpuip.h"
#ifdef __APPLE__
#define cimg_OS 0
#endif
#include "3rdparty/CImg.h"
#include <boost/numpy.hpp>
#include <boost/python.hpp>
#include <ImfRgbaFile.h>
#include <ImfConvert.h>
//----------------------------------------------------------------------------//
namespace np = boost::numpy;
namespace bp = boost::python;
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
namespace io {
//----------------------------------------------------------------------------//
template<typename T>
void _CImgToNumpy(np::ndarray & data,
                  const unsigned int channels,
                  const std::string & filename)
{
    cimg_library::CImg<T> image(filename.c_str());
    const unsigned int w = image.width();
    const unsigned int h = image.height();
    const unsigned int c = image.spectrum();
    const unsigned int stride = w * h;

    // allocate numpy array
    const np::dtype dtype = sizeof(T) == 1 ?
            np::dtype::get_builtin<unsigned char>() :
            np::dtype::get_builtin<float>();
    data = np::zeros(boost::python::make_tuple(w,h,channels), dtype);
    
    T * to = reinterpret_cast<T*>(data.get_data());
    const T * from = image.data();
        
    // two common cases have their 3rd for loop unrolled
    if (channels == 1) { 
        for(unsigned int i = 0; i < w; ++i) {
            for(unsigned int j = 0; j < h; ++j) {
                to[(i + w * j)] = from[i + w*j];
            }
        }
    } else if (channels == 3 and c == 3) {
        for(unsigned int i = 0; i < w; ++i) {
            for(unsigned int j = 0; j < h; ++j) {
                to[3*(i+w*j)  ] = from[i + w*j           ];
                to[3*(i+w*j)+1] = from[i + w*j + stride  ];
                to[3*(i+w*j)+2] = from[i + w*j + 2*stride];
            }
        }
    } else {
        for(unsigned int i = 0; i < w; ++i) {
            for(unsigned int j = 0; j < h; ++j) {
                for(unsigned int k = 0; k < (c<channels ? c : channels); ++k) {
                    to[channels*(i + w * j)+k] = from[stride*k+ i + w*j];
                }
            }
        }
    }
}
//----------------------------------------------------------------------------//
template<typename T>
void _NumpyToCImg(const np::ndarray & data,
                  const unsigned int channels,
                  const std::string & filename)
{
    cimg_library::CImg<T> image(data.shape(0), data.shape(1), 1,data.shape(2));
    const int w = image.width();
    const int h = image.height();
    const int c = image.spectrum();
    const int stride = w * h;
    T * to = image.data();
    const T * from = reinterpret_cast<T*>(data.get_data());
    // two common cases have their 3rd for loop unrolled
    if (channels == 1) { 
        for(int i = 0; i < w; ++i) {
            for(int j = 0; j < h; ++j) {
                to[(i + w * j)] = from[i + w*j];
            }
        }
    } else if (channels == 3 and c == 3) {
        for(int i = 0; i < w; ++i) {
            for(int j = 0; j < h; ++j) {
                to[i + w*j           ] = from[3*(i+w*j)  ];
                to[i + w*j + stride  ] = from[3*(i+w*j)+1];
                to[i + w*j + 2*stride] = from[3*(i+w*j)+2];
            }
        }
    } else {
        for(int i = 0; i < w; ++i) {
            for(int j = 0; j < h; ++j) {
                for(int k = 0; k < c; ++k) {
                    to[stride*k+ i + w*j] = from[channels*(i + w * j)+k];
                }
            }
        }
    }
    image.save(filename.c_str());
}
//----------------------------------------------------------------------------//
template<typename T_FLOAT, int T_CHANNELS>
void _ConvertToHalf(const T_FLOAT * in,
                    std::vector<Imf::Rgba> & out,
                    unsigned int width,
                    unsigned int height)                    
{
    for(unsigned int i = 0; i < width; ++i) {
        for (unsigned int j = 0; j < height; ++j) {
            const size_t idx = i + j * width;
            if (T_CHANNELS >= 1) {
                out[idx].r = in[T_CHANNELS*idx];
            }
            if (T_CHANNELS >= 2) {
                out[idx].g = in[T_CHANNELS*idx+1];
            }
            if (T_CHANNELS >= 3) {
                out[idx].b = in[T_CHANNELS*idx+2];
            }
            if (T_CHANNELS >= 4) {
                out[idx].a = in[T_CHANNELS*idx+3];
            }
        }
    }
}
//----------------------------------------------------------------------------//
template<typename T_FLOAT>
Imf::RgbaChannels
_ConvertToExr(const T_FLOAT * data,
              std::vector<Imf::Rgba> & out,
              unsigned int width,
              unsigned int height,
              unsigned int channels)
{
    using namespace Imf;
    RgbaChannels format;
    switch(channels) {
        case 1:
            _ConvertToHalf<T_FLOAT,1>(data, out, width, height);
            format = WRITE_Y;
            break;
        case 2:
            _ConvertToHalf<T_FLOAT,2>(data, out, width, height);
            format = WRITE_YC;
            break;
        case 3:
            _ConvertToHalf<T_FLOAT,3>(data, out, width, height);
            format = WRITE_RGB;
            break;
        case 4:
            _ConvertToHalf<T_FLOAT,4>(data, out, width, height);
            format = WRITE_RGBA;
            break;
        default:
            format = WRITE_RGBA; 
    }
    return format;
}
//----------------------------------------------------------------------------//
void _NumpyToHalfExr(const boost::numpy::ndarray & data,
                     unsigned int channels,
                     const std::string & filename,
                     int numThreads)
{
    using namespace Imf;

    unsigned int width = data.shape(0);
    unsigned int height = data.shape(1);

    setGlobalThreadCount(numThreads);
    
    Rgba * data_ptr;
    std::vector<Rgba> halfdata;
    RgbaChannels format = WRITE_RGBA;
    if (channels==4 and data.get_dtype() == np::detail::get_float_dtype<16>()) {
        data_ptr = reinterpret_cast<Rgba*>(data.get_data());
    } else {
        halfdata.reserve(width*height);
        data_ptr = halfdata.data();

        if (data.get_dtype() == np::detail::get_float_dtype<16>()) {
            format = _ConvertToExr(reinterpret_cast<half*>(data.get_data()),
                                   halfdata, width, height, channels);
        } else if (data.get_dtype() == np::detail::get_float_dtype<32>()) {
            format = _ConvertToExr(reinterpret_cast<float*>(data.get_data()),
                                   halfdata, width, height, channels);
        }
    }
    RgbaOutputFile file(filename.c_str(), width, height, format);
    file.setFrameBuffer (data_ptr, 1, width);
    file.writePixels(height);
}
//----------------------------------------------------------------------------//
template<typename T_FLOAT, int T_CHANNELS>
void _ConvertFromHalf(const Imf::Rgba * in,
                      T_FLOAT * out,
                      unsigned int width,
                      unsigned int height)                    
{
    for(unsigned int i = 0; i < width; ++i) {
        for (unsigned int j = 0; j < height; ++j) {
            const size_t idx = i + j * width;
            if (T_CHANNELS >= 1) {
                out[T_CHANNELS*idx] = in[idx].r;
            }
            if (T_CHANNELS >= 2) {
                out[T_CHANNELS*idx+1] = in[idx].g;
            }
            if (T_CHANNELS >= 3) {
                out[T_CHANNELS*idx+2] = in[idx].b;
            }
            if (T_CHANNELS >= 4) {
                out[T_CHANNELS*idx+3] = in[idx].a;
            }
        }
    }
}
//----------------------------------------------------------------------------//
template<typename T_FLOAT>
void
_ConvertFromExr(const Imf::Rgba * in,
                T_FLOAT * out,
                unsigned int width,
                unsigned int height,
                unsigned int channels)
{
    using namespace Imf;
    switch(channels) {
        case 1:
            _ConvertFromHalf<T_FLOAT,1>(in, out, width, height);
            break;
        case 2:
            _ConvertFromHalf<T_FLOAT,2>(in, out, width, height);
            break;
        case 3:
            _ConvertFromHalf<T_FLOAT,3>(in, out, width, height);
            break;
        case 4:
            _ConvertFromHalf<T_FLOAT,4>(in, out, width, height);
            break;
    }
}
//----------------------------------------------------------------------------//
void _HalfExrToNumpy(boost::numpy::ndarray & data,
                     const std::string & filename,
                     const gpuip::Buffer & buffer,
                     int numThreads)
{
    using namespace Imf;
    setGlobalThreadCount(numThreads);

    RgbaInputFile file(filename.c_str());
    Imath::Box2i dw = file.dataWindow();
    unsigned int width = dw.max.x - dw.min.x + 1;
    unsigned int height = dw.max.y - dw.min.y + 1;
    if (buffer.type == gpuip::Buffer::HALF && buffer.channels == 4) {
        data = np::zeros(boost::python::make_tuple(width,height,4),
                         np::detail::get_float_dtype<16>());
        file.setFrameBuffer(reinterpret_cast<Rgba*>(data.get_data()), 1, width);
        file.readPixels (dw.min.y, dw.max.y);
    } else {
        std::vector<Rgba> halfdata(width*height);
        file.setFrameBuffer(halfdata.data(), 1, width);
        file.readPixels(dw.min.y, dw.max.y);

        if (buffer.type == gpuip::Buffer::HALF) {
            data = np::zeros(bp::make_tuple(width,height,buffer.channels),
                             np::detail::get_float_dtype<16>());
            _ConvertFromExr<half>(halfdata.data(),
                                  reinterpret_cast<half*>(data.get_data()),
                                  width, height, buffer.channels);
        } else if (buffer.type == gpuip::Buffer::FLOAT) {
            data = np::zeros(bp::make_tuple(width,height,buffer.channels),
                             np::detail::get_float_dtype<32>());
            _ConvertFromExr<float>(halfdata.data(),
                                   reinterpret_cast<float*>(data.get_data()),
                                   width, height, buffer.channels);
        }
    }
}
//----------------------------------------------------------------------------//
void ReadFromFile(boost::numpy::ndarray * npyarray,
                  const Buffer & buffer,
                  const std::string & filename,
                  int numThreads)
{
    switch(buffer.type) {
        case Buffer::UNSIGNED_BYTE:
            _CImgToNumpy<unsigned char>(*npyarray, buffer.channels, filename);
            break;
        case Buffer::HALF:
            _HalfExrToNumpy(*npyarray, filename, buffer, numThreads);
            break;
        case Buffer::FLOAT:
            _HalfExrToNumpy(*npyarray, filename, buffer, numThreads);
            break;
    }
}
//----------------------------------------------------------------------------//
void WriteToFile(const boost::numpy::ndarray * npyarray,
                 const Buffer & buffer,
                 const std::string & filename,
                 int numThreads)
{
    switch(buffer.type) {
        case Buffer::UNSIGNED_BYTE:
            _NumpyToCImg<unsigned char>(*npyarray, buffer.channels, filename);
            break;
        case Buffer::HALF:
            _NumpyToHalfExr(*npyarray, buffer.channels, filename, numThreads);
            break;
        case Buffer::FLOAT:
            _NumpyToHalfExr(*npyarray, buffer.channels, filename, numThreads);
            break;
    }
}
//----------------------------------------------------------------------------//
} // end namespace io
} // end namespace gpuip
//----------------------------------------------------------------------------//

