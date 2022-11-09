#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../libs/stb_image.h"
#include "../libs/stb_image_write.h"
#include "../libs/utils.h"
#include <chrono>
#include <stdio.h>
#include <string>
using namespace std;

void sharpen(unsigned char *, int, int, int, unsigned char *);

int main(int argc, const char *argv[])
{
    if (argc != 2)
        return 0;

    string base = argv[1];
    string fin = "input/" + base;
    string fout = "output/" + base;

    int w, h, ch;
    unsigned char *image = stbi_load(fin.c_str(), &w, &h, &ch, 0);
    unsigned char *gs = new unsigned char[w * h * ch];

    long start = timeMillis();
    sharpen(image, w, h, ch, gs);
    long end = timeMillis();

    if (fout.find("jpg") != -1)
        stbi_write_jpg(fout.c_str(), w, h, ch, gs, 100);
    else
        stbi_write_png(fout.c_str(), w, h, ch, gs, w * ch);

    printf("it took %ld ms", end - start);

    delete[] image, gs;
    return 0;
}

void sharpen(unsigned char *in, int w, int h, int c, unsigned char *out)
{
    int size = w * h * c;

    for (int i = 0; i < size; ++i)
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
