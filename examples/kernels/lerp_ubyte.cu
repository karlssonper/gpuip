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
    const uchar4 aa = a[idx];
    const uchar4 bb = b[idx];

    out[idx] = (uchar4)((1-alpha)*aa.x + alpha*bb.x,
                           (1-alpha)*aa.y + alpha*bb.y,
                        (1-alpha)*aa.z + alpha*bb.z,
                        255);
}