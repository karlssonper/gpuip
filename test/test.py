import pygpuip as gpuip
import numpy

opencl_codeA = """
 __kernel void
my_kernelA(__global const float * A,
           __global float * B,
           __global float * C,
           const int incA,
           const float incB,
           const int width,
           const int height)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // array index
    const int idx = x + width * y;

    // inside image bounds check
    if (x >= width || y >= height) {
        return;
    }

    // kernel code
    B[idx] = A[idx] + incA *0.1;
    C[idx] = A[idx] + incB;
}
"""
opencl_codeB = """
__kernel void
my_kernelB(__global const float * B,
           __global const float * C,
           __global float * A,
           const int width,
           const int height)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // array index
    const int idx = x + width * y;

    // inside image bounds check
    if (x >= width || y >= height) {
        return;
    }

    // kernel code
    A[idx] =  B[idx] + C[idx];
}
"""
opencl_boilerplateA = \
"""__kernel void
my_kernelA(__global const float * A,
           __global float * B,
           __global float * C,
           const int incA,
           const float incB,
           const int width,
           const int height)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // array index
    const int idx = x + width * y;

    // inside image bounds check
    if (x >= width || y >= height) {
        return;
    }

    // kernel code
    B[idx] = 0;
    C[idx] = 0;
}"""
opencl_boilerplateB = \
"""__kernel void
my_kernelB(__global const float * B,
           __global const float * C,
           __global float * A,
           const int width,
           const int height)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // array index
    const int idx = x + width * y;

    // inside image bounds check
    if (x >= width || y >= height) {
        return;
    }

    // kernel code
    A[idx] = 0;
}"""

cuda_codeA = """
__global__ void
my_kernelA(float * A,
           float * B,
           float * C,
           const int incA,
           const float incB,
           const int width,
           const int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // array index
    const int idx = x + width * y;

    // inside image bounds check
    if (x >= width || y >= height) {
        return;
    }

    // kernel code
    B[idx] = A[idx] + incA * 0.1;
    C[idx] = A[idx] + incB;
}"""
cuda_codeB = """
__global__ void
my_kernelB(float * B,
           float * C,
           float * A,
           const int width,
           const int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // array index
    const int idx = x + width * y;

    // inside image bounds check
    if (x >= width || y >= height) {
        return;
    }

    // kernel code
    A[idx] = B[idx] + C[idx];
}"""

cuda_boilerplateA = \
"""__global__ void
my_kernelA(const float * A,
           float * B,
           float * C,
           const int incA,
           const float incB,
           const int width,
           const int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // array index
    const int idx = x + width * y;

    // inside image bounds check
    if (x >= width || y >= height) {
        return;
    }

    // kernel code
    B[idx] = 0;
    C[idx] = 0;
}"""
cuda_boilerplateB = \
"""__global__ void
my_kernelB(const float * B,
           const float * C,
           float * A,
           const int width,
           const int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // array index
    const int idx = x + width * y;

    // inside image bounds check
    if (x >= width || y >= height) {
        return;
    }

    // kernel code
    A[idx] = 0;
}"""

glsl_codeA = """
#version 120
uniform sampler2D A;
uniform int incA;
uniform float incB;
varying vec2 x; // texture coordinates
uniform float dx; // delta
void main()
{
    gl_FragData[0] = vec4(texture2D(A, x).x+incA*0.1,0,0,1);
    gl_FragData[1] = vec4(texture2D(A, x).x+incB,0,0,1);
}"""
glsl_codeB = """
#version 120
uniform sampler2D B;
uniform sampler2D C;
varying vec2 x; // texture coordinates
uniform float dx; // delta
void main()
{
    gl_FragData[0] = vec4(texture2D(B, x).x +
                          texture2D(C, x).x, 0, 0, 1);
}"""

glsl_boilerplateA = \
"""#version 120
uniform sampler2D A;
uniform int incA;
uniform float incB;
varying vec2 x; // texture coordinates
uniform float dx; // delta
void main()
{
    // gl_FragData[0] is buffer B
    glFragData[0] = vec4(0,0,0,1);

    // gl_FragData[1] is buffer C
    glFragData[1] = vec4(0,0,0,1);
}"""
glsl_boilerplateB = \
"""#version 120
uniform sampler2D B;
uniform sampler2D C;
varying vec2 x; // texture coordinates
uniform float dx; // delta
void main()
{
    // gl_FragData[0] is buffer A
    glFragData[0] = vec4(0,0,0,1);
}"""

width = 4
height = 4 
N = width * height
no_error = ""

def test(env, codeA, codeB, boilerplateA, boilerplateB):
    base = gpuip.gpuip(env)
    base.SetDimensions(width, height)
    assert base

    buffers = [gpuip.Buffer() for i in range(3)]
    for i, b in enumerate(buffers):
        b.name = "b%i" % i
        b.data = numpy.zeros((width,height), dtype = numpy.float32)
        b.channels = 1
        b.bpp = 4
        base.AddBuffer(b)

    kernelA = base.CreateKernel("my_kernelA")
    assert kernelA 
    assert kernelA.name == "my_kernelA" 
    kernelA.code = codeA
    kernelA.SetInBuffer("A", buffers[0])
    kernelA.SetOutBuffer("B", buffers[1])
    kernelA.SetOutBuffer("C", buffers[2])

    incA = gpuip.ParamInt()
    incA.name = "incA"
    incA.value = 2
    kernelA.SetParam(incA)

    incB = gpuip.ParamFloat()
    incB.name = "incB"
    incB.value = 0.25
    kernelA.SetParam(incB)
    assert base.GetBoilerplateCode(kernelA) == boilerplateA

    kernelB = base.CreateKernel("my_kernelB")
    assert kernelB
    assert kernelB.name == "my_kernelB" 
    kernelB.code = codeB
    kernelB.SetInBuffer("B", buffers[1])
    kernelB.SetInBuffer("C", buffers[2])
    kernelB.SetOutBuffer("A", buffers[0])
    assert base.GetBoilerplateCode(kernelB) == boilerplateB

    assert base.InitBuffers() == no_error
    assert base.InitBuffers() == no_error # reinit should not break things
    indata = numpy.zeros((width,height), dtype = numpy.float32)
    for i in range(width):
        for j in range(height):
            indata[i][j] = i + j * width
    buffers[0].data[:] = indata
    assert base.WriteBuffer(buffers[0]) == no_error

    assert base.Build() == no_error
    assert base.Build() == no_error # rebuilding should not break things
    assert base.Process() == no_error

    for b in buffers:
        assert base.ReadBuffer(b) == no_error

    def eq(a,b):
        print a,b
        return abs(a-b) < 0.0001
    
    b0,b1,b2 = buffers[0].data, buffers[1].data, buffers[2].data
    for i in range(width):
        for j in range(height):
            assert eq(b1[i][j], indata[i][j] + incA.value*0.1)
            assert eq(b2[i][j], indata[i][j] + incB.value)

            assert eq(b0[i][j], b1[i][j] + b2[i][j])
    print "Test passed!"

if __name__ == '__main__':
    if gpuip.CanCreateGpuEnvironment(gpuip.Environment.OpenCL):
        print "Testing OpenCL..." 
        test(gpuip.Environment.OpenCL, opencl_codeA, opencl_codeB,
             opencl_boilerplateA, opencl_boilerplateB)

    if gpuip.CanCreateGpuEnvironment(gpuip.Environment.CUDA):
        print "Testing CUDA..." 
        test(gpuip.Environment.CUDA, cuda_codeA, cuda_codeB,
             cuda_boilerplateA, cuda_boilerplateB)

    if gpuip.CanCreateGpuEnvironment(gpuip.Environment.GLSL):
        print "Testing GLSL..." 
        test(gpuip.Environment.GLSL, glsl_codeA, glsl_codeB,
             glsl_boilerplateA, glsl_boilerplateB)
