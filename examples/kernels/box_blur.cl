float4 read(__global const half * in_half, int x, int y, int w)
{
    const int idx = x + w * y;
    return (float4)(vload_half(4 * idx + 0, in_half),
                    vload_half(4 * idx + 1, in_half),
                    vload_half(4 * idx + 2, in_half),
                    vload_half(4 * idx + 3, in_half));
}

__kernel void
box_blur(__global const half * in_half,
         __global half * out_half,
         const int n,
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
    float4 out = (float4)(0, 0, 0, 0);
    int count = 0;
    for(int j = y - n; j <= y + n; ++j)  {
        for(int i = x - n; i <= x+n; ++i) {
            if (i>=0 && j>= 0 && i < width && j < height) {
                out += read(in_half, i, j, width);
                count += 1;
            }
        }
    }
    out /= count;

    // float to half conversion
    vstore_half(out.x, 4 * idx + 0, out_half);
    vstore_half(out.y, 4 * idx + 1, out_half);
    vstore_half(out.z, 4 * idx + 2, out_half);
    vstore_half(out.w, 4 * idx + 3, out_half);
}