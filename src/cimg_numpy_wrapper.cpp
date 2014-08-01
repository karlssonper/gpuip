#include "cimg_numpy_wrapper.h"
#include "3rdparty/CImg.h"
//----------------------------------------------------------------------------//
namespace np = boost::numpy;
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
template<typename T>
void _CImgToNumpy(np::ndarray & data,
                  const unsigned int channels,
                  const std::string & filename)
{
    cimg_library::CImg<T> image(filename.c_str());
    const int w = image.width();
    const int h = image.height();
    const int c = image.spectrum();
    const int stride = w * h;

    // allocate numpy array
    const np::dtype dtype = sizeof(T) == 1 ?
            np::dtype::get_builtin<unsigned char>() :
            np::dtype::get_builtin<float>();
    data = np::zeros(boost::python::make_tuple(w,h,channels), dtype);
    
    T * to = reinterpret_cast<T*>(data.get_data());
    const T * from = image.data();
        
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
                to[3*(i+w*j)  ] = from[i + w*j           ];
                to[3*(i+w*j)+1] = from[i + w*j + stride  ];
                to[3*(i+w*j)+2] = from[i + w*j + 2*stride];
            }
        }
    } else {
        for(int i = 0; i < w; ++i) {
            for(int j = 0; j < h; ++j) {
                for(int k = 0; k < c; ++k) {
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
#define _EXPLICIT_TEMPLATE_INSTANTIATION_TYPE(TYPE)                     \
    template                                                            \
    void _CImgToNumpy<TYPE>(np::ndarray &,                              \
                             const unsigned int,                        \
                            const std::string &);                       \
    template                                                            \
    void _NumpyToCImg<TYPE>(const np::ndarray & data,                   \
                            const unsigned int channels,                \
                            const std::string & filename);
_EXPLICIT_TEMPLATE_INSTANTIATION_TYPE(float);
_EXPLICIT_TEMPLATE_INSTANTIATION_TYPE(unsigned char);
//----------------------------------------------------------------------------//
} // end namespace
//----------------------------------------------------------------------------//

