#include "util.hpp"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

#define HEADER_PATH_ID 1
#define DEM_PATH_ID 2
#define SOURCE_PATH_ID 3
#define OUTPUT_PATH_ID 4
#define STEPS_ID 5

#define P_R 0.5
#define P_EPSILON 0.001
#define SIZE_OF_X 5
#define STRLEN 256

#define BLOCK_SIZE 16

#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value)                              \
  ((M)[(((n) * (rows) * (columns)) + ((i) * (columns)) + (j))] = (value))
#define BUF_GET(M, rows, columns, n, i, j)                                     \
  (M[(((n) * (rows) * (columns)) + ((i) * (columns)) + (j))])

void readHeaderInfo(char *path, int &nrows, int &ncols, double &nodata) {
  FILE *f;

  if ((f = fopen(path, "r")) == 0) {
    printf("%s configuration header file not found\n", path);
    exit(0);
  }

  char str[STRLEN];
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str);
  ncols = atoi(str);
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str);
  nrows = atoi(str);
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str);
  nodata = atof(str);
}

bool loadGrid2D(double *M, int rows, int columns, char *path) {
  FILE *f = fopen(path, "r");

  if (!f) {
    printf("%s grid file not found\n", path);
    exit(0);
  }

  char str[STRLEN];
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < columns; j++) {
      fscanf(f, "%s", str);
      SET(M, columns, i, j, atof(str));
    }

  fclose(f);

  return true;
}

bool saveGrid2Dr(double *M, int rows, int columns, char *path) {
  FILE *f;
  f = fopen(path, "w");

  if (!f)
    return false;

  char str[STRLEN];
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      sprintf(str, "%f ", GET(M, columns, i, j));
      fprintf(f, "%s ", str);
    }
    fprintf(f, "\n");
  }

  fclose(f);

  return true;
}

double *addLayer2D(int rows, int columns) {
  double *tmp = (double *)malloc(sizeof(double) * rows * columns);
  if (!tmp)
    return NULL;
  return tmp;
}

void simulationInit(int i, int j, int r, int c, double *Sz, double *Sh) {
  double z, h;
  h = GET(Sh, c, i, j);

  if (h > 0.0) {
    z = GET(Sz, c, i, j);
    SET(Sz, c, i, j, z - h);
  }
}

__global__ void resetFlowsParallelized(int r, int c, double *Sf) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= 0 && i < r && j >= 0 && j < c) {
    BUF_SET(Sf, r, c, 0, i, j, 0.0);
    BUF_SET(Sf, r, c, 1, i, j, 0.0);
    BUF_SET(Sf, r, c, 2, i, j, 0.0);
    BUF_SET(Sf, r, c, 3, i, j, 0.0);
  }
}

__global__ void flowsComputationParallelized(int r, int c, double nodata,
                                             int *Xi, int *Xj, double *Sz,
                                             double *Sh, double *Sf, double p_r,
                                             double p_epsilon, int i_start,
                                             int i_end, int j_start,
                                             int j_end) {
  bool eliminated_cells[5] = {false, false, false, false, false};
  bool again;
  int cells_count;
  double average;
  double m;
  double u[5];
  int n;
  double z, h;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < i_start || i >= i_end || j < j_start || j >= j_end)
    return;

  m = GET(Sh, c, i, j) - p_epsilon;
  u[0] = GET(Sz, c, i, j) + p_epsilon;
  z = GET(Sz, c, i + Xi[1], j + Xj[1]);
  h = GET(Sh, c, i + Xi[1], j + Xj[1]);
  u[1] = z + h;
  z = GET(Sz, c, i + Xi[2], j + Xj[2]);
  h = GET(Sh, c, i + Xi[2], j + Xj[2]);
  u[2] = z + h;
  z = GET(Sz, c, i + Xi[3], j + Xj[3]);
  h = GET(Sh, c, i + Xi[3], j + Xj[3]);
  u[3] = z + h;
  z = GET(Sz, c, i + Xi[4], j + Xj[4]);
  h = GET(Sh, c, i + Xi[4], j + Xj[4]);
  u[4] = z + h;

  do {
    again = false;
    average = m;
    cells_count = 0;

    for (n = 0; n < 5; n++)
      if (!eliminated_cells[n]) {
        average += u[n];
        cells_count++;
      }

    if (cells_count != 0)
      average /= cells_count;

    for (n = 0; n < 5; n++)
      if ((average <= u[n]) && (!eliminated_cells[n])) {
        eliminated_cells[n] = true;
        again = true;
      }
  } while (again);

  if (!eliminated_cells[1])
    BUF_SET(Sf, r, c, 0, i, j, (average - u[1]) * p_r);
  if (!eliminated_cells[2])
    BUF_SET(Sf, r, c, 1, i, j, (average - u[2]) * p_r);
  if (!eliminated_cells[3])
    BUF_SET(Sf, r, c, 2, i, j, (average - u[3]) * p_r);
  if (!eliminated_cells[4])
    BUF_SET(Sf, r, c, 3, i, j, (average - u[4]) * p_r);
}

__global__ void widthUpdateParallelized(int r, int c, double nodata, int *Xi,
                                        int *Xj, double *Sz, double *Sh,
                                        double *Sf, int i_start, int i_end,
                                        int j_start, int j_end) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < i_start || i >= i_end || j < j_start || j >= j_end)
    return;

  double h_next;
  h_next = GET(Sh, c, i, j);
  h_next +=
      BUF_GET(Sf, r, c, 3, i + Xi[1], j + Xj[1]) - BUF_GET(Sf, r, c, 0, i, j);
  h_next +=
      BUF_GET(Sf, r, c, 2, i + Xi[2], j + Xj[2]) - BUF_GET(Sf, r, c, 1, i, j);
  h_next +=
      BUF_GET(Sf, r, c, 1, i + Xi[3], j + Xj[3]) - BUF_GET(Sf, r, c, 2, i, j);
  h_next +=
      BUF_GET(Sf, r, c, 0, i + Xi[4], j + Xj[4]) - BUF_GET(Sf, r, c, 3, i, j);

  SET(Sh, c, i, j, h_next);
}

int main(int argc, char **argv) {
  int rows, cols;
  double nodata;
  readHeaderInfo(argv[HEADER_PATH_ID], rows, cols, nodata);

  int r = rows;
  int c = cols;
  int i_start = 1, i_end = r - 1;
  int j_start = 1, j_end = c - 1;
  double *Sz;
  double *Sh;
  double *Sf;
  int *Xi = new int[5];
  int *Xj = new int[5];
  double p_r = P_R;
  double p_epsilon = P_EPSILON;
  int steps = atoi(argv[STEPS_ID]);

  Sz = addLayer2D(r, c);
  Sh = addLayer2D(r, c);
  Sf = addLayer2D(SIZE_OF_X * r, c);

  int *Xi_device;
  int *Xj_device;

  cudaMalloc(&Xi_device, sizeof(int) * 5);
  cudaMalloc(&Xj_device, sizeof(int) * 5);

  Xi[0] = 0;
  Xi[1] = -1;
  Xi[2] = 0;
  Xi[3] = 0;
  Xi[4] = 1;

  Xj[0] = 0;
  Xj[1] = 0;
  Xj[2] = -1;
  Xj[3] = 1;
  Xj[4] = 0;

  cudaMemcpy(Xi_device, Xi, sizeof(int) * 5, cudaMemcpyHostToDevice);
  cudaMemcpy(Xj_device, Xj, sizeof(int) * 5, cudaMemcpyHostToDevice);

  loadGrid2D(Sz, r, c, argv[DEM_PATH_ID]);
  loadGrid2D(Sh, r, c, argv[SOURCE_PATH_ID]);

  for (int i = 0; i < r; i++)
    for (int j = 0; j < c; j++)
      simulationInit(i, j, r, c, Sz, Sh);
  double *Sz_device;
  cudaMalloc(&Sz_device, sizeof(double) * r * c);
  double *Sh_device;
  cudaMalloc(&Sh_device, sizeof(double) * r * c);
  double *Sf_device;
  cudaMalloc(&Sf_device, sizeof(double) * r * c * SIZE_OF_X);

  cudaMemcpy(Sz_device, Sz, sizeof(double) * r * c, cudaMemcpyHostToDevice);
  cudaMemcpy(Sh_device, Sh, sizeof(double) * r * c, cudaMemcpyHostToDevice);
  cudaMemcpy(Sf_device, Sf, sizeof(double) * r * c * SIZE_OF_X,
             cudaMemcpyHostToDevice);

  util::Timer cl_timer;

  double dimB = 8.0f;
  dim3 dimGrid(ceil(r / dimB), ceil(c / dimB), 1);
  dim3 dimBlock(dimB, dimB, 1);

  printf("colonne: %d\n", c);
  printf("Grid: %dx%d\n", dimGrid.x, dimGrid.y);
  printf("Block: %dx%d\n", dimBlock.x, dimBlock.y);

  for (int s = 0; s < steps; ++s) {
    resetFlowsParallelized<<<dimGrid, dimBlock>>>(r, c, Sf_device);
    cudaDeviceSynchronize();

    flowsComputationParallelized<<<dimGrid, dimBlock>>>(
        r, c, nodata, Xi_device, Xj_device, Sz_device, Sh_device, Sf_device,
        p_r, p_epsilon, i_start, i_end, j_start, j_end);
    cudaDeviceSynchronize();

    widthUpdateParallelized<<<dimGrid, dimBlock>>>(
        r, c, nodata, Xi_device, Xj_device, Sz_device, Sh_device, Sf_device,
        i_start, i_end, j_start, j_end);
    cudaDeviceSynchronize();
  }
  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  cudaMemcpy(Sh, Sh_device, sizeof(double) * r * c, cudaMemcpyDeviceToHost);
  printf("Elapsed time: %lf [s]\n", cl_time);
  saveGrid2Dr(Sh, r, c, argv[OUTPUT_PATH_ID]);

  printf("Releasing memory...\n");
  cudaFree(Sh_device);
  cudaFree(Sz_device);
  cudaFree(Sf_device);
  cudaFree(Xi_device);
  cudaFree(Xj_device);
  delete[] Sz;
  delete[] Sh;
  delete[] Sf;
  delete[] Xi;
  delete[] Xj;

  return 0;
}