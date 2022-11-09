#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../libs/stb_image_write.h"
#include "../libs/utils.h"
#include <cstdlib>
#include <stdio.h>
#include <string>
using namespace std;

const int N = 100;
const int COLS = 2 << 10;
const int ROWS = 2 << 10;

__global__ void init(double *, double *, int, int);
__global__ void transition(double *, int, int, double *);

void linearize_jpg(double *, unsigned char *);
void save_jpg(unsigned char *, int);

int main(int argc, char const *argv[])
{
    // create 2 matrices of size WxH and initialize them to -22.0
    double *current;
    double *next;

    cudaMallocManaged(&current, ROWS * COLS * sizeof(double));
    cudaMallocManaged(&next, ROWS * COLS * sizeof(double));

    unsigned char *pic;
    cudaMallocManaged(&pic, ROWS * COLS * 3 * sizeof(unsigned char));

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(ROWS / threadsPerBlock.x + 1, COLS / threadsPerBlock.y + 1);

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
        cudaDeviceSynchronize();

        if (i % 10 == 0)
        {
            linearize_jpg(current, pic);
            save_jpg(pic, f++);
        }

        // swap the matrices
        temp = current;
        current = next;
        next = temp;
    }

    long end = timeMillis() - start;
    printf("Time: %ld ms\nConverting and cleaning up...", end);

    // execute a cmd command to convert the images to a video
    system("ffmpeg -hwaccel cuda -r 120 -i output\/\%d.jpg -c:v h264_nvenc -b:v 5M output\/heat.mp4 -y && .\\output\\heat.mp4");
    // system("del output\\*.jpg");

    cudaFree(current);
    cudaFree(next);
    cudaFree(pic);
    return 0;
}

// function that converts the array linear array to a 3 channel jpg
void linearize_jpg(double *current, unsigned char *pic)
{
    // find min and max in the array
    double min = -22;
    double max = 22;

    // normalize the array from 0 to 255
    for (int i = 0; i < ROWS * COLS; i++)
    {
        pic[i * 3] = (current[i] - min) / (max - min) * 255;
        pic[i * 3 + 1] = 0;
        pic[i * 3 + 2] = 0;
    }
}

// function to save the input array as a jpg image using stbi_write_jpg
void save_jpg(unsigned char *pic, int frame_num)
{
    string filename = "output/" + to_string(frame_num) + ".jpg";
    stbi_write_jpg(filename.c_str(), COLS, ROWS, 3, pic, 100);
}

__global__ void init(double *m1, double *m2, int rows, int cols)
{
    // initialize the matrices to -22.0
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols)
    {
        m1[i * rows + j] = -22.0;
        m2[i * rows + j] = -22.0;
    }
}

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

// function called transition that takes 2 matrices as input
void transition(double **current, double **next)
{
    // starting from the 10th row apply the heat transfer equation
    for (int i = 10; i < ROWS - 10; i++)
    {
        for (int j = 1; j < COLS - 2; j++)
        {
            // sum the cardinal directions in a cumulative variable and multiply by 4
            double sum = 0;
            sum += current[i][j - 1];
            sum += current[i][j + 1];
            sum += current[i - 1][j];
            sum += current[i + 1][j];
            sum *= 4;

            // sum the diagonal directions in the sum variable and divide by 20
            sum += current[i - 1][j - 1];
            sum += current[i - 1][j + 1];
            sum += current[i + 1][j - 1];
            sum += current[i + 1][j + 1];
            sum /= 20;

            // set the current cell in the new matrix to the sum
            next[i][j] = sum;
        }
    }
}

// function that creates a 1D array of size WxH copying the values from the input matrix
void linearize_jpg(double **input, unsigned char *out)
{
    // find the min and max values in the matrix
    double min = input[0][0];
    double max = input[0][0];
    for (int i = 0; i < ROWS; i++)
    {
        for (int j = 0; j < COLS; j++)
        {
            if (input[i][j] < min)
                min = input[i][j];
            if (input[i][j] > max)
                max = input[i][j];
        }
    }

    // normalize the values in the matrix in the out array
    for (int i = 0; i < ROWS; i++)
    {
        for (int j = 0; j < COLS; j++)
        {
            out[(i * COLS + j) * 3] = static_cast<unsigned char>((input[i][j] - min) / (max - min) * 255);
            out[(i * COLS + j) * 3 + 1] = 0;
            out[(i * COLS + j) * 3 + 2] = 0;
        }
    }
}