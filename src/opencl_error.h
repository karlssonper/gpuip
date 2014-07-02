#ifndef GPUIP_OPENCL_ERROR_H_
#define GPUIP_OPENCL_ERROR_H_
//----------------------------------------------------------------------------//
namespace gpuip {
//----------------------------------------------------------------------------//
inline bool _clErrorInitBuffers(cl_int cl_err, std::string * err)
{
    if (cl_err != CL_SUCCESS) {
        (*err) += "OpenCL: error when creating buffers\n";
        switch(cl_err) {
            //TODO: add cases here
            default:
                break;
        }
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
inline bool _clErrorCreateProgram(cl_int cl_err, std::string * err)
{
    if (cl_err != CL_SUCCESS) {
        (*err) += "OpenCL: error when creating program\n";
        switch(cl_err) {
            //TODO: add cases here
            default:
                break;
        }
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
inline bool _clErrorBuildProgram(cl_int cl_err, std::string * err,
                                 cl_program p, cl_device_id device_id,
                                 std::string kernel_name)
{
    if (cl_err != CL_SUCCESS) {
        (*err) += "OpenCL: error when building kernel: ";
        (*err) += kernel_name;
        (*err) += "\n\n";
        char buf[0x10000];
        clGetProgramBuildInfo(p, device_id, CL_PROGRAM_BUILD_LOG,
                              0x10000, buf, NULL);
        (*err) += std::string(buf);
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
inline bool _clErrorCreateKernel(cl_int cl_err, std::string * err)
{
    if (cl_err != CL_SUCCESS) {
        (*err) += "OpenCL: error when creating kernel\n";
        switch(cl_err) {
            //TODO: add cases here
            default:
                break;
        }
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
inline bool _clErrorCopy(cl_int cl_err, std::string * err,
                         const std::string & buffer, Buffer::CopyOperation op)
{
    if (cl_err != CL_SUCCESS) {
        (*err) += "OpenCL: error when copying data ";
        (*err) += op == Buffer::READ_DATA ? "FROM" : "TO";
        (*err) += " buffer ";
        (*err) += buffer;
        switch(cl_err) {
            //TODO: add cases here
            default:
                break;
        }
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
inline bool _clErrorSetKernelArg(cl_int cl_err, std::string * err,
                                 const std::string & kernel_name)
{
    if (cl_err != CL_SUCCESS) {
        (*err) += "OpenCL: error in argument setup in kernel ";
        (*err) += kernel_name;
        switch(cl_err) {
            //TODO: add cases here
            default:
                break;
        }
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
inline bool _clErrorEnqueueKernel(cl_int cl_err, std::string * err,
                                  const std::string & kernel_name)
{
    if (cl_err != CL_SUCCESS) {
        (*err) += "OpenCL: error when enqueuing kernel ";
        (*err) += kernel_name;
        switch(cl_err) {
            //TODO: add cases here
            default:
                break;
        }
        return true;
    }
    return false;
}
//----------------------------------------------------------------------------//
} // end namespace gpuip
//----------------------------------------------------------------------------//
#endif
