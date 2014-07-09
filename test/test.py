import pyGpuip as gpuip
import numpy

opencl_codeA = """
__kernel void my_kernelA(__global float * A,
                         __global float * B,
                         __global float * C,
                         int incA,
                         float incB)
{
    int x = get_global_id(0); int y = get_global_id(1);
    B[x+y*4] =  A[x+y*4] + incA *0.1;
    C[x+y*4] =  A[x+y*4] + incB;
}
"""
opencl_codeB = """
__kernel void my_kernelB(__global float * B,
                         __global float * C,
                         __global float * A)
{
    int x = get_global_id(0); int y = get_global_id(1);
    A[x+y*4] =  B[x+y*4] + C[x+y*4];
}
"""

cuda_codeA = """
__global__ void my_kernelA(float * A,
                           float * B,
                           float * C,
                           int incA,
                           float incB)
{
    if (threadIdx.x < 4 && threadIdx.y < 4) {
        int idx = threadIdx.x + 4 * threadIdx.y;
        B[idx] = A[idx] + incA * 0.1;
        C[idx] = A[idx] + incB;
    }
}"""
cuda_codeB = """
__global__ void my_kernelB(float * B,
                           float * C,
                           float * A)
{
    if (threadIdx.x < 4 && threadIdx.y < 4) {
        int idx = threadIdx.x + 4 * threadIdx.y;
        A[idx] = B[idx] + C[idx];
    }
}"""

glsl_codeA = """
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
uniform sampler2D B;
uniform sampler2D C;
varying vec2 texcoord;
void main()
{
    gl_FragData[0] = vec4(texture2D(B, texcoord).x +
                          texture2D(C, texcoord).x, 0, 0, 1);
}"""

width = 4
height = 4 
N = width * height
no_error = ""

def test(env, codeA, codeB):
    base = gpuip.gpuip(env, width, height)
    assert base

    buffers = [gpuip.Buffer() for i in range(3)]
    buffers[0].name = "A"
    buffers[1].name = "B"
    buffers[2].name = "C"
    
    for b in buffers:
        b.channels = 1
        b.bpp = 4
        base.AddBuffer(b)

    kernelA = base.CreateKernel("my_kernelA")
    assert kernelA 
    assert kernelA.name == "my_kernelA" 
    kernelA.code = codeA
    kernelA.AddInBuffer(buffers[0])
    kernelA.AddOutBuffer(buffers[1])
    kernelA.AddOutBuffer(buffers[2])

    incA = gpuip.ParamInt()
    incA.name = "incA"
    incA.value = 2
    kernelA.AddParamInt(incA)

    incB = gpuip.ParamFloat()
    incB.name = "incB"
    incB.value = 0.25
    kernelA.AddParamFloat(incB)

    kernelB = base.CreateKernel("my_kernelB")
    assert kernelB
    assert kernelB.name == "my_kernelB" 
    kernelB.code = codeB
    kernelB.AddInBuffer(buffers[1])
    kernelB.AddInBuffer(buffers[2])
    kernelB.AddOutBuffer(buffers[0])

    assert base.InitBuffers() == no_error
    bufferA_indata = numpy.zeros((width,height), dtype = numpy.float32)
    for i in range(width):
        for j in range(height):
            bufferA_indata[i][j] = i + j * width
    assert base.WriteBuffer(buffers[0], bufferA_indata) == no_error
    assert base.Build() == no_error
    assert base.Process() == no_error

    outdata = [numpy.zeros((width,height), dtype = numpy.float32) \
                          for i in range(3)]
    for i in range(3):
        assert base.ReadBuffer(buffers[i], outdata[i]) == no_error

    def eq(a,b):
        return abs(a-b) < 0.0001

    for i in range(width):
        for j in range(height):
            assert eq(outdata[1][i][j], bufferA_indata[i][j] + incA.value*0.1)
            assert eq(outdata[2][i][j], bufferA_indata[i][j] + incB.value)

            assert eq(outdata[0][i][j], outdata[1][i][j] + outdata[2][i][j])

if __name__ == '__main__':
    test(gpuip.Environment.OpenCL, opencl_codeA, opencl_codeB)
    test(gpuip.Environment.CUDA, cuda_codeA, cuda_codeB)
    test(gpuip.Environment.GLSL, glsl_codeA, glsl_codeB)
