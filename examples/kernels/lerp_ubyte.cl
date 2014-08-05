__kernel void
lerp(__global const uchar4 * a,
     __global const uchar4 * b,
     __global uchar4 * out,
     const float alpha,
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
    const float4 af = convert_float4(a[idx]);
    const float4 bf = convert_float4(b[idx]);
    uchar4 tmp = convert_uchar4((1-alpha)*af+alpha*bf);

    // set alpha
    tmp.w = 255;

    out[idx] = tmp;
}