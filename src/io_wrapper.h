#ifndef GPUIP_IO_WRAPPER_H_
#define GPUIP_IO_WRAPPER_H_
//----------------------------------------------------------------------------//
#include <string>
//----------------------------------------------------------------------------//
namespace boost { namespace numpy { class ndarray; } }
namespace gpuip {
struct Buffer;
//----------------------------------------------------------------------------//
namespace io {
//----------------------------------------------------------------------------//
void ReadFromFile(boost::numpy::ndarray * npyarray,
                  const Buffer & buffer,
                  const std::string & filename,
                  int numThreads = 0);
//----------------------------------------------------------------------------//
void WriteToFile(const boost::numpy::ndarray * npyarray,
                 const Buffer & buffer,
                 const std::string & filename,
                 int numThreads = 0);
//----------------------------------------------------------------------------//
} // end namespace io
} // end namespace gpuip
//----------------------------------------------------------------------------//
#endif
