__global__ void
lerp(const unsigned short * a_half,
     const unsigned short * b_half,
     unsigned short * out_half,
     const float alpha,
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

    // half to float conversion
    const float4 a = make_float4(__half2float(a_half[4 * idx + 0]),
                                 __half2float(a_half[4 * idx + 1]),
                                 __half2float(a_half[4 * idx + 2]),
                                 __half2float(a_half[4 * idx + 3]));
    const float4 b = make_float4(__half2float(b_half[4 * idx + 0]),
                                 __half2float(b_half[4 * idx + 1]),
                                 __half2float(b_half[4 * idx + 2]),
                                 __half2float(b_half[4 * idx + 3]));

    // kernel code
    float4 out = (1-alpha) * a + alpha * b;

    // float to half conversion
    out_half[4 * idx + 0] = __float2half_rn(out.x);
    out_half[4 * idx + 1] = __float2half_rn(out.y);
    out_half[4 * idx + 2] = __float2half_rn(out.z);
    out_half[4 * idx + 3] = __float2half_rn(out.w);
}