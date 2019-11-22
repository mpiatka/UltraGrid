#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <time.h>

        __global__
void kern_RGBtoRGBA(unsigned char *dst,
                size_t dstPitch,
                unsigned char *src,
                size_t srcPitch,
                size_t width,
                size_t height)
{
        const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
        const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

        if(x >= width)
                return;

        if(y >= height)
                return;

        uchar3 *src_px = (uchar3 *) (src + y * srcPitch) + x;
        uchar4 *dst_px = (uchar4 *) (dst + y * dstPitch) + x;

        const uchar3 px = *src_px;

        *dst_px = make_uchar4(px.x, px.y, px.z, 0);
}

void cuda_RGBto_RGBA(unsigned char *dst,
                size_t dstPitch,
                unsigned char *src,
                size_t srcPitch,
                size_t width,
                size_t height,
                CUstream_st *stream){

        dim3 blockSize(32,32);
        dim3 numBlocks((width + blockSize.x - 1) / blockSize.x,
                        (height + blockSize.y - 1) / blockSize.y);

        kern_RGBtoRGBA<<<numBlocks, blockSize, 0, stream>>>(dst, dstPitch, src, srcPitch, width, height);
}
