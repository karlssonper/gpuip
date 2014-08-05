__kernel void
lerp(__global const float4 * a,
     __global const float4 * b,
     __global float4 * out,
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
    out[idx] = (1-alpha) * a[idx] + alpha * b[idx];
}