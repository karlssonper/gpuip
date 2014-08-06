__device__ float4 read(const uchar4 * in, int x, int y, int width)
{
    const uchar4 v = in[x + y * width];
    return make_float4(v.x, v.y, v.z, v.w);
}

__device__ float4 compute_sobelx(const uchar4 * in, int x, int y, int w)
{
    return -2*read(in,x-1,y-1,w) - read(in,x-1,y,w) - read(in,x-1,y+1,w)
            +2*read(in,x+1,y-1,w) + read(in,x+1,y,w) + read(in,x+1,y+1,w);
}

__device__ float4 compute_sobely(const uchar4 * in, int x, int y, int w)
{
    return -2*read(in,x-1,y-1,w) - read(in,x,y-1,w) - read(in,x+1,y-1,w)
            +2*read(in,x-1,y+1,w) + read(in,x,y+1,w) + read(in,x+1,y+1,w);
}

__global__ void
sobel(const uchar4 * in,
      uchar4 * sobelx,
      uchar4 * sobely,
      uchar4 * gradient,
      uchar4 * edges,
      const float primary_treshold,
      const float secondary_treshold,
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
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        sobelx[idx] = make_uchar4(0,0,0,255);
        sobely[idx] = make_uchar4(0,0,0,255);
        gradient[idx] = make_uchar4(0,0,0,255);
        edges[idx] = make_uchar4(0,0,0,255);
        return;
    }

    float4 sx = compute_sobelx(in,x,y,width);
    float4 sy = compute_sobely(in,x,y,width);
    
    sobelx[idx] = make_uchar4(abs(sx.x), abs(sx.y), abs(sx.z), 255);
    sobely[idx] = make_uchar4(abs(sy.x), abs(sy.y), abs(sy.z), 255);

    float grad = (abs(sx.x)+abs(sx.y)+abs(sx.z)+
                  abs(sy.x)+abs(sy.y)+abs(sy.z))/3.0;
    gradient[idx] = make_uchar4(grad, grad, grad, 255);

    const bool prim_edge = grad > primary_treshold;
    const bool sec_edge = grad > secondary_treshold;
    const unsigned char edge = prim_edge * 255 | sec_edge * 125;
    edges[idx] = make_uchar4(edge,edge,edge,255);
}
