#ifndef GPUIP_CIMG_NUMPY_WRAPPER_H_
#define GPUIP_CIMG_NUMPY_WRAPPER_H_
//----------------------------------------------------------------------------//
#include <boost/numpy.hpp>
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
template<typename T>
void _CImgToNumpy(boost::numpy::ndarray & data,
                         const unsigned int channels,
                         const std::string & filename);
//----------------------------------------------------------------------------//
template<typename T>
void _NumpyToCImg(const boost::numpy::ndarray & data,
                         const unsigned int channels,
                         const std::string & filename);
//----------------------------------------------------------------------------//
} // end namespace gpuip
//----------------------------------------------------------------------------//
#endif
