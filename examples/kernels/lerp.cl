__kernel void
lerp(__global const half * a_half,
     __global const half * b_half,
     __global half * out_half,
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

    // half to float conversion
    const float4 a = (float4)(vload_half(4 * idx + 0, a_half),
                              vload_half(4 * idx + 1, a_half),
                              vload_half(4 * idx + 2, a_half),
                              vload_half(4 * idx + 3, a_half));
    const float4 b = (float4)(vload_half(4 * idx + 0, b_half),
                              vload_half(4 * idx + 1, b_half),
                              vload_half(4 * idx + 2, b_half),
                              vload_half(4 * idx + 3, b_half));

    // kernel code
    float4 out = (1-alpha) * a + alpha * b;

    // float to half conversion
    vstore_half(out.x, 4 * idx + 0, out_half);
    vstore_half(out.y, 4 * idx + 1, out_half);
    vstore_half(out.z, 4 * idx + 2, out_half);
    vstore_half(out.w, 4 * idx + 3, out_half);
}