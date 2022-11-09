#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../libs/stb_image.h"
#include "../libs/stb_image_write.h"
#include <string>
using namespace std;

__global__ void bw(unsigned char *image, int w, int h, int c, unsigned char *out);
__global__ void sharpen(unsigned char *image, int w, int h, int c, unsigned char *out);
__global__ void sharpen2D(unsigned char *image, int w, int h, int c, unsigned char *out);

int main(int argc, const char *argv[])
{
    if (argc != 2)
        return 0;

    string base = argv[1];
    string fin = "input/" + base;
    string fout = "output/" + base;

    int width, height, channels;
    auto image = stbi_load(fin.c_str(), &width, &height, &channels, 0);
    unsigned char *out;
    unsigned char *dimg;

    auto bytesize = sizeof(unsigned char) * width * height * channels;

    cudaMallocManaged(&dimg, bytesize);
    cudaMallocManaged(&out, bytesize);
    cudaMemcpy(dimg, image, bytesize, cudaMemcpyHostToDevice);

    dim3 blocksize(16, 16);
    dim3 blocks(ceil((width * channels) / blocksize.x) + 1, ceil((height * channels) / blocksize.y) + 1);

    sharpen2D<<<blocks, blocksize>>>(dimg, width, height, channels, out);
    cudaDeviceSynchronize();

    if (fout.find("jpg") != -1)
        stbi_write_jpg(fout.c_str(), width, height, channels, out, 100);
    else
        stbi_write_png(fout.c_str(), width, height, channels, out, width * channels);

    cudaFree(dimg);
    cudaFree(out);
    delete[] image, dimg, out;

    return 0;
}

__global__ void bw(unsigned char *image, int w, int h, int c, unsigned char *out)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    int i = (row * w * c) + col;

    if (col < (w * c) && row < h)
    {
        auto v = (image[i] + image[i + 1] + image[i + 2]) / 3;
        out[i] = out[i + 1] = out[i + 2] = v;
        if (c == 4)
            out[i + 3] = image[i + 3];
    }
}

__global__ void sharpen(unsigned char *in, int w, int h, int c, unsigned char *out)
{
    int size = w * h * c;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size)
    {
        int v = 0;
        if ((i - c) % (w * c) >= 0)
        {
            // ovest
            v -= in[i - c];
        }
        else
            v -= in[i];

        if ((i + c) % (w * c) != 0)
        {
            // est
            v -= in[i + c];
        }
        else
            v -= in[i];

        if (i - (w * c) >= 0)
        {
            // north
            v -= in[i - (w * c)];
        }
        else
            v -= in[i];

        if (i + (w * c) <= w * h * c)
        {
            // south
            v -= in[i + (w * c)];
        }
        else
            v -= in[i];

        v += in[i] * 5;

        if (v > 255)
            v = 255;
        if (v < 0)
            v = 0;

        out[i] = v;
    }
}

__global__ void sharpen2D(unsigned char *in, int w, int h, int c, unsigned char *out)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    int i = (row * w * c) + col;

    if (col < (w * c) && row < h)
    {
        int v = 0;
        if ((i - c) % (w * c) >= 0)
        {
            // ovest
            v -= in[i - c];
        }
        else
            v -= in[i];

        if ((i + c) % (w * c) != 0)
        {
            // est
            v -= in[i + c];
        }
        else
            v -= in[i];

        if (i - (w * c) >= 0)
        {
            // north
            v -= in[i - (w * c)];
        }
        else
            v -= in[i];

        if (i + (w * c) <= w * h * c)
        {
            // south
            v -= in[i + (w * c)];
        }
        else
            v -= in[i];

        v += in[i] * 5;

        if (v > 255)
            v = 255;
        if (v < 0)
            v = 0;

        out[i] = v;
    }
}