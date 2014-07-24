import pygpuip as gpuip
import numpy

opencl_codeA = """
__kernel void
my_kernelA(__global const float * A,
           __global float * B,
           __global float * C,
           int incA,
           float incB,
           int width,
           int height)
{
    int x = get_global_id(0); int y = get_global_id(1);
    B[x+y*4] =  A[x+y*4] + incA *0.1;
    C[x+y*4] =  A[x+y*4] + incB;
}
"""
opencl_codeB = """
__kernel void
my_kernelB(__global const float * B,
           __global const float * C,
           __global float * A,
           int width,
           int height)
{
    int x = get_global_id(0); int y = get_global_id(1);
    A[x+y*4] =  B[x+y*4] + C[x+y*4];
}
"""
opencl_boilerplateA = \
"""__kernel void
my_kernelA(__global const float * A,
           __global float * B,
           __global float * C,
           int incA,
           float incB,
           int width,
           int height)
{
}"""
opencl_boilerplateB = \
"""__kernel void
my_kernelB(__global const float * B,
           __global const float * C,
           __global float * A,
           int width,
           int height)
{
}"""

cuda_codeA = """
__global__ void
my_kernelA(float * A,
           float * B,
           float * C,
           int incA,
           float incB,
           int width,
           int height)
{
    if (threadIdx.x < 4 && threadIdx.y < 4) {
        int idx = threadIdx.x + 4 * threadIdx.y;
        B[idx] = A[idx] + incA * 0.1;
        C[idx] = A[idx] + incB;
    }
}"""
cuda_codeB = """
__global__ void
my_kernelB(float * B,
           float * C,
           float * A,
           int width,
           int height)
{
    if (threadIdx.x < 4 && threadIdx.y < 4) {
        int idx = threadIdx.x + 4 * threadIdx.y;
        A[idx] = B[idx] + C[idx];
    }
}"""

cuda_boilerplateA = \
"""__global__ void
my_kernelA(const float * A,
           float * B,
           float * C,
           int incA,
           float incB,
           int width,
           int height)
{
}"""
cuda_boilerplateB = \
"""__global__ void
my_kernelB(const float * B,
           const float * C,
           float * A,
           int width,
           int height)
{
}"""

glsl_codeA = """
#version 120
uniform sampler2D A;
uniform int incA;
uniform float incB;
varying vec2 texcoord;
void main()
{
    gl_FragData[0] = vec4(texture2D(A, texcoord).x+incA*0.1,0,0,1);
    gl_FragData[1] = vec4(texture2D(A, texcoord).x+incB,0,0,1);
}"""
glsl_codeB = """
#version 120
uniform sampler2D B;
uniform sampler2D C;
varying vec2 texcoord;
void main()
{
    gl_FragData[0] = vec4(texture2D(B, texcoord).x +
                          texture2D(C, texcoord).x, 0, 0, 1);
}"""

glsl_boilerplateA = \
"""#version 120
uniform sampler2D A;
uniform int incA;
uniform float incB;
varying vec2 texcoord;
void main()
{
}"""
glsl_boilerplateB = \
"""#version 120
uniform sampler2D B;
uniform sampler2D C;
varying vec2 texcoord;
void main()
{
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
    indata = numpy.zeros((width,height), dtype = numpy.float32)
    for i in range(width):
        for j in range(height):
            indata[i][j] = i + j * width
    buffers[0].data[:] = indata
    assert base.WriteBuffer(buffers[0]) == no_error

    assert base.Build() == no_error
    assert base.Process() == no_error

    for b in buffers:
        assert base.ReadBuffer(b) == no_error

    def eq(a,b):
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
