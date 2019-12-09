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

#define max(a, b)      (((a) > (b))? (a): (b))
#define min(a, b)      (((a) < (b))? (a): (b))

__global__
void kern_UYVYtoRGBA(unsigned char *dst,
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

	const int uyvy_block = x / 2;

	uchar4 *src_px = (uchar4 *) (src + y * srcPitch) + uyvy_block;
	uchar4 *dst_px = (uchar4 *) (dst + y * dstPitch) + x;

	const uchar4 block = *src_px;

	int u = block.x;
	int v = block.z;
	int luma = (x % 2) ? block.w : block.y;

	int r = min(max(1.164f*(luma - 16) + 1.793f*(v - 128), 0), 255);
	int g = min(max(1.164f*(luma - 16) - 0.534f*(v - 128) - 0.213f*(u - 128), 0), 255); 
	int b = min(max(1.164f*(luma - 16) + 2.115f*(u - 128), 0), 255);

	*dst_px = make_uchar4(r, g, b, 0);
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

void cuda_UYVY_to_RGBA(unsigned char *dst,
                size_t dstPitch,
                unsigned char *src,
                size_t srcPitch,
                size_t width,
                size_t height,
                CUstream_st *stream){

        dim3 blockSize(32,32);
        dim3 numBlocks((width + blockSize.x - 1) / blockSize.x,
                        (height + blockSize.y - 1) / blockSize.y);

        kern_UYVYtoRGBA<<<numBlocks, blockSize, 0, stream>>>(dst, dstPitch, src, srcPitch, width, height);
}
