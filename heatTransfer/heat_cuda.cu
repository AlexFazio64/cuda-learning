#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../libs/stb_image_write.h"
#include "../libs/utils.h"
#include <cstdlib>
#include <stdio.h>
#include <string>
using namespace std;

const int N = 20000;
const int ROWS = 2 << 10;
const int COLS = 2 << 10;

__global__ void init(double *, double *, int, int);
__global__ void transition(double *, int, int, double *);
__global__ void linearize_jpg(double *, int, int, unsigned char *);

void save_jpg(unsigned char *, int);

int main(int argc, char const *argv[])
{
    bool cache = argc > 1;
    // create 2 matrices of size WxH and initialize them to -22.0
    double *current;
    double *next;

    cudaMallocManaged(&current, ROWS * COLS * sizeof(double));
    cudaMallocManaged(&next, ROWS * COLS * sizeof(double));

    // create an array to store the linearized image
    unsigned char *pic;
    cudaMallocManaged(&pic, ROWS * COLS * 3 * sizeof(unsigned char));

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(ceil(COLS / threadsPerBlock.x) + 1, ceil((ROWS) / threadsPerBlock.y) + 1);

    // initialize the matrices in a kernel
    init<<<numBlocks, threadsPerBlock>>>(current, next, ROWS, COLS);
    cudaDeviceSynchronize();

    // set the first and last 10 rows to 22 on both matrices
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < COLS; j++)
        {
            current[i * ROWS + j] = 22.0;
            next[i * ROWS + j] = 22.0;
            current[(ROWS - 1 - i) * ROWS + j] = 22.0;
            next[(ROWS - 1 - i) * ROWS + j] = 22.0;
        }
    }

    double *temp;
    int f = 0;
    long start = timeMillis();

    // transition between the 2 matrices N times
    for (int i = 0; i < N; i++)
    {
        transition<<<numBlocks, threadsPerBlock>>>(current, ROWS, COLS, next);

        if (i % 50 == 0 && cache)
        {
            linearize_jpg<<<numBlocks, threadsPerBlock>>>(current, ROWS, COLS, pic);
            cudaDeviceSynchronize();
            save_jpg(pic, f++);
            printf("saved frame %d\n", f - 1);
        }
        cudaDeviceSynchronize();

        // swap the matrices
        temp = current;
        current = next;
        next = temp;
    }

    long end = timeMillis() - start;
    printf("Time: %ld ms\nConverting and cleaning up...\n", end);

    // execute a cmd command to convert the images to a video
    if (cache)
    {
        system("ffmpeg -hwaccel cuda -r 60 -i output\/\%d.jpg -c:v h264_nvenc -b:v 5M output\/heat.mp4 -y && .\\output\\heat.mp4");
        system("del output\\*.jpg");
    }
    
    cudaFree(current);
    cudaFree(next);
    cudaFree(pic);

    delete[] current, next, pic;
    return 0;
}

// function that converts the array linear array to a 3 channel jpg
__global__ void linearize_jpg(double *current, int rows, int cols, unsigned char *pic)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // find min and max in the array
    const double min = -22.0;
    const double max = 22.0;

    if (c < cols && r < rows)
    {
        // normalize the value to 0-255
        double val = (current[r * cols + c] - min) / (max - min) * 255;
        pic[(r * cols + c) * 3] = val;
    }
}

// function that initializes the matrices to -22.0
__global__ void init(double *m1, double *m2, int rows, int cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols)
    {
        m1[i * rows + j] = -22.0;
        m2[i * rows + j] = -22.0;
    }
}

// state transition function
__global__ void transition(double *curr, int rows, int cols, double *out)
{
    // get the current thread's index
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    // if the thread is inside the matrix
    if (c < cols - 1 && c > 0 && r < (rows - 10) && r > 9)
    {
        // sum the cardinal directions in a cumulative variable and multiply by 4
        double sum = 0;
        sum += curr[r * rows + c - 1];
        sum += curr[r * rows + c + 1];
        sum += curr[(r - 1) * rows + c];
        sum += curr[(r + 1) * rows + c];
        sum *= 4;

        // sum the diagonal directions in the sum variable and divide by 20
        sum += curr[(r - 1) * rows + c - 1];
        sum += curr[(r - 1) * rows + c + 1];
        sum += curr[(r + 1) * rows + c - 1];
        sum += curr[(r + 1) * rows + c + 1];
        sum /= 20;

        // set the current cell in the new matrix to the sum
        out[r * rows + c] = sum;
    }
}

void save_jpg(unsigned char *pic, int frame_num)
{
    string filename = "output/" + to_string(frame_num) + ".jpg";
    stbi_write_jpg(filename.c_str(), COLS, ROWS, 3, pic, 100);
}