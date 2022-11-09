#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../libs/stb_image_write.h"
#include "../libs/utils.h"
#include <cstdlib>
#include <stdio.h>
#include <string>
using namespace std;

const int N = 4000;
const int COLS = 200;
const int ROWS = 200;

void transition(double **, double **);
void linearize_jpg(double **, unsigned char *);
void save_jpg(unsigned char *, int);

int main(int argc, char const *argv[])
{
    // create 2 matrices of size WxH and initialize them to -22.0
    unsigned char *pic = new unsigned char[ROWS * COLS * 3];
    double **current = new double *[ROWS];
    double **next = new double *[ROWS];

    for (int i = 0; i < ROWS; i++)
    {
        current[i] = new double[COLS];
        next[i] = new double[COLS];

        for (int j = 0; j < COLS; j++)
        {
            current[i][j] = -22.0;
            next[i][j] = -22.0;
        }
    }

    // set the first and last 10 rows to 22 on both matrices
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < COLS; j++)
        {
            current[i][j] = 22.0;
            next[i][j] = 22.0;
            current[ROWS - i - 1][j] = 22.0;
            next[ROWS - i - 1][j] = 22.0;
        }
    }

    double **temp;
    int f = 0;
    long start = timeMillis();

    // transition between the 2 matrices N times
    for (int i = 0; i < N; i++)
    {
        transition(current, next);
        if (i % 5 == 0)
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
    system("del output\\*.jpg");

    // delete both matrices
    for (int i = 0; i < COLS; i++)
    {
        delete[] current[i];
        delete[] next[i];
    }

    delete[] current, next, pic;
    return 0;
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
            out[(i * COLS + j) * 3] = static_cast<unsigned char>(input[i][j] - min) / (max - min) * 255;
            out[(i * COLS + j) * 3 + 1] = 0;
            out[(i * COLS + j) * 3 + 2] = 0;
        }
    }
}

// function to save the input array as a jpg image using stbi_write_jpg
void save_jpg(unsigned char *pic, int frame_num)
{
    string filename = "output/" + to_string(frame_num) + ".jpg";
    stbi_write_jpg(filename.c_str(), COLS, ROWS, 3, pic, 100);
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