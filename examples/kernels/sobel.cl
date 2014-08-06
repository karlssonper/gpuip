float4 read(__global const uchar4 * in, int x, int y, int width)
{
    return convert_float4(in[x + width * y]);
}

float4 compute_sobelx(__global const uchar4 * in, int x, int y, int w)
{
    return -2*read(in,x-1,y-1,w) - read(in,x-1,y,w) - read(in,x-1,y+1,w)
            +2*read(in,x+1,y-1,w) + read(in,x+1,y,w) + read(in,x+1,y+1,w);
}

float4 compute_sobely(__global const uchar4 * in, int x, int y, int w)
{
    return -2*read(in,x-1,y-1,w) - read(in,x,y-1,w) - read(in,x+1,y-1,w)
            +2*read(in,x-1,y+1,w) + read(in,x,y+1,w) + read(in,x+1,y+1,w);
}

__kernel void
sobel(__global const uchar4 * in,
      __global uchar4 * sobelx,
      __global uchar4 * sobely,
      __global uchar4 * gradient,
      __global uchar4 * edges,
      float primary_treshold,
      float secondary_treshold,
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
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        sobelx[idx] = (uchar4)(0, 0, 0, 255);
        sobely[idx] = (uchar4)(0, 0, 0, 255);
        gradient[idx] = (uchar4)(0, 0, 0, 255);
        edges[idx] = (uchar4)(0, 0, 0, 255);
        return;
    }

    const float4 sx = compute_sobelx(in, x, y, width);
    const float4 sy = compute_sobely(in, x, y, width);
    const float grad = (fabs(sx.x)+fabs(sx.y)+fabs(sx.z)+
                        fabs(sy.x)+fabs(sy.y)+fabs(sy.z))/3.0;

    sobelx[idx] = convert_uchar4(sx);
    sobely[idx] = convert_uchar4(sy);
    
    const unsigned char gradu = convert_uchar(grad);
    gradient[idx] = (uchar4)(gradu, gradu, gradu, 255);

    const bool prim_edge = grad > primary_treshold;
    const bool sec_edge = grad > secondary_treshold;
    const uchar edge = 255 * prim_edge | 125 * sec_edge;

    edges[idx] = (uchar4)(edge, edge, edge, 255);
}
