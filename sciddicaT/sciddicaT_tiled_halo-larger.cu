#include "util.hpp"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define HEADER_PATH_ID 1
#define DEM_PATH_ID 2
#define SOURCE_PATH_ID 3
#define OUTPUT_PATH_ID 4
#define STEPS_ID 5

#define P_R 0.5
#define P_EPSILON 0.001
#define SIZE_OF_X 5
#define STRLEN 256

#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value)                              \
  ((M)[(((n) * (rows) * (columns)) + ((i) * (columns)) + (j))] = (value))
#define BUF_GET(M, rows, columns, n, i, j)                                     \
  (M[(((n) * (rows) * (columns)) + ((i) * (columns)) + (j))])

#define TILE_WIDTH 8

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

void simulationInit(int i, int j, int r, int c, double *Sz_h, double *Sh_h) {
  double z, h;
  h = GET(Sh_h, c, i, j);

  if (h > 0.0) {
    z = GET(Sz_h, c, i, j);
    SET(Sz_h, c, i, j, z - h);
  }
}

__global__ void resetFlowsParallelized(int r, int c, int i_start, int i_end,
                                       double *Sf_h) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= i_start && i < i_end && j >= 1 && j < c) {
    BUF_SET(Sf_h, r, c, 0, i, j, 0.0);
    BUF_SET(Sf_h, r, c, 1, i, j, 0.0);
    BUF_SET(Sf_h, r, c, 2, i, j, 0.0);
    BUF_SET(Sf_h, r, c, 3, i, j, 0.0);
  }
}

__global__ void flowsComputationParallelized(int r, int c, int i_start,
                                             int i_end, double *Sz_h,
                                             double *Sh_h, double *Sf_h,
                                             double *halo_Sz, double *halo_Sh,
                                             double p_r, double p_epsilon) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= i_start && i < i_end && j >= 1 && j < c) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ double Sz_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ double Sh_shared[TILE_WIDTH][TILE_WIDTH];
    
    Sz_shared[ty][tx] = GET(Sz_h, c, i, j);
    Sh_shared[ty][tx] = GET(Sh_h, c, i, j);
    __syncthreads();

    bool eliminated_cells[5] = {false, false, false, false, false};
    bool again;
    int cells_count;
    double average;
    double m;
    double u[5];
    int n;
    double z, h;

    m = Sh_shared[ty][tx] - p_epsilon;
    u[0] = Sz_shared[ty][tx] + p_epsilon;

    if (i == 0) {
      z = halo_Sz[j];
      h = halo_Sh[j];
    } else {
      if (ty > 0) {
        z = Sz_shared[ty - 1][tx];
        h = Sh_shared[ty - 1][tx];
      } else {
        z = GET(Sz_h, c, i - 1, j);
        h = GET(Sh_h, c, i - 1, j);
      }
    }
    u[1] = z + h;

    if (tx > 0) {
      z = Sz_shared[ty][tx - 1];
      h = Sh_shared[ty][tx - 1];
    } else {
      z = GET(Sz_h, c, i, j - 1);
      h = GET(Sh_h, c, i, j - 1);
    }
    u[2] = z + h;

    if (tx <= blockDim.x - 2) {
      z = Sz_shared[ty][tx + 1];
      h = Sh_shared[ty][tx + 1];
    } else {
      z = GET(Sz_h, c, i, j + 1);
      h = GET(Sh_h, c, i, j + 1);
    }
    u[3] = z + h;

    if (i == r / 2 - 1) {
      z = halo_Sz[j];
      h = halo_Sh[j];
    } else {
      if (ty <= blockDim.y - 2) {
        z = Sz_shared[ty + 1][tx];
        h = Sh_shared[ty + 1][tx];
      } else {
        z = GET(Sz_h, c, i + 1, j);
        h = GET(Sh_h, c, i + 1, j);
      }
    }
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
      BUF_SET(Sf_h, r, c, 0, i, j, (average - u[1]) * p_r);
    if (!eliminated_cells[2])
      BUF_SET(Sf_h, r, c, 1, i, j, (average - u[2]) * p_r);
    if (!eliminated_cells[3])
      BUF_SET(Sf_h, r, c, 2, i, j, (average - u[3]) * p_r);
    if (!eliminated_cells[4])
      BUF_SET(Sf_h, r, c, 3, i, j, (average - u[4]) * p_r);
  }
}

__global__ void widthUpdateParallelized(int r, int c, int i_start, int i_end,
                                        double *Sh_h, double *Sf_h) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int i_offset = i;

  __shared__ double Sf_shared[TILE_WIDTH][TILE_WIDTH][SIZE_OF_X];
  Sf_shared[ty][tx][0] = BUF_GET(Sf_h, r, c, 0, i_offset, j);
  Sf_shared[ty][tx][1] = BUF_GET(Sf_h, r, c, 1, i_offset, j);
  Sf_shared[ty][tx][2] = BUF_GET(Sf_h, r, c, 2, i_offset, j);
  Sf_shared[ty][tx][3] = BUF_GET(Sf_h, r, c, 3, i_offset, j);
  __syncthreads();

  if (i >= i_start && i < i_end && j >= 1 && j < c) {
    double h_next;
    h_next = GET(Sh_h, c, i, j);

    double b0 = Sf_shared[ty][tx][0];
    double b1 = Sf_shared[ty][tx][1];
    double b2 = Sf_shared[ty][tx][2];
    double b3 = Sf_shared[ty][tx][3];

    double bUp;
    if (ty > 0) {
      bUp = Sf_shared[ty - 1][tx][3];
    } else if (ty == 0) {
      bUp = BUF_GET(Sf_h, r, c, 3, i_offset - 1, j);
    }

    double bLeft;
    if (tx > 0) {
      bLeft = Sf_shared[ty][tx - 1][2];
    } else if (tx == 0) {
      bLeft = BUF_GET(Sf_h, r, c, 2, i_offset, j - 1);
    }

    double bRight;
    if (tx <= blockDim.x - 2) {
      bRight = Sf_shared[ty][tx + 1][1];
    } else if (tx == blockDim.x - 1) {
      bRight = BUF_GET(Sf_h, r, c, 1, i_offset, j + 1);
    }

    double bDown;
    if (ty <= blockDim.y - 2) {
      bDown = Sf_shared[ty + 1][tx][0];
    } else if (ty == blockDim.y - 1) {
      bDown = BUF_GET(Sf_h, r, c, 0, i_offset + 1, j);
    }

    h_next += bUp - b0;
    h_next += bLeft - b1;
    h_next += bRight - b2;
    h_next += bDown - b3;

    SET(Sh_h, c, i, j, h_next);
  }
}

int main(int argc, char **argv) {
  int rows, cols;
  double nodata;
  readHeaderInfo(argv[HEADER_PATH_ID], rows, cols, nodata);

  int r = rows;
  int c = cols;
  double p_r = P_R;
  double p_epsilon = P_EPSILON;
  int steps = atoi(argv[STEPS_ID]);

  double *Sz_h;
  double *Sh_h;
  double *halo_Sz;
  double *halo_Sh;
  double *Sz_d;
  double *Sh_d;
  double *Sf_d;

  Sz_h = addLayer2D(r, c);
  Sh_h = addLayer2D(r, c);
  printf("Dimensione Sz/Sh: %dx%d\n", r, c);
  printf("Dimensione Sf: %dx%d\n", SIZE_OF_X * r, c);

  loadGrid2D(Sz_h, r, c, argv[DEM_PATH_ID]);
  loadGrid2D(Sh_h, r, c, argv[SOURCE_PATH_ID]);
  for (int i = 0; i < r; i++)
    for (int j = 0; j < c; j++)
      simulationInit(i, j, r, c, Sz_h, Sh_h);

  cudaMallocManaged(&Sz_d, sizeof(double) * r * c);
  cudaMallocManaged(&Sh_d, sizeof(double) * r * c);
  cudaMallocManaged(&Sf_d, sizeof(double) * r * c * SIZE_OF_X);
  cudaMallocManaged(&halo_Sz, c * sizeof(double));
  cudaMallocManaged(&halo_Sh, c * sizeof(double));

  int i_start, i_end;
  i_start = 1;
  i_end = r - 1;

  util::Timer cl_timer;

  double dimB = 8.0f;
  dim3 dimGrid(ceil(c / dimB), ceil(r / dimB), 1);
  dim3 dimBlock(dimB, dimB, 1);

  printf("colonne: %d\n", c);
  printf("Grid: %dx%d\n", dimGrid.x, dimGrid.y);
  printf("Block: %dx%d\n", dimBlock.x, dimBlock.y);

  for (int s = 0; s < steps; ++s) {
    resetFlowsParallelized<<<dimGrid, dimBlock>>>(i_start, i_end, r, c, Sf_d);
    cudaDeviceSynchronize();

    flowsComputationParallelized<<<dimGrid, dimBlock>>>(
        r, c, i_start, i_end, Sz_d, Sh_d, Sf_d, halo_Sh, halo_Sz, p_r,
        p_epsilon);
    cudaDeviceSynchronize();

    widthUpdateParallelized<<<dimGrid, dimBlock>>>(r, c, i_start, i_end, Sh_d,
                                                   Sf_d);
    cudaDeviceSynchronize();
  }
  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  cudaMemcpy(Sh_h, Sh_d, sizeof(double) * r * c, cudaMemcpyDeviceToHost);

  printf("Elapsed time: %lf [s]\n", cl_time);
  saveGrid2Dr(Sh_h, r, c, argv[OUTPUT_PATH_ID]);
  printf("Releasing memory...\n");

  cudaFree(Sh_d);
  cudaFree(Sz_d);
  cudaFree(Sf_d);
  cudaFree(halo_Sz);
  cudaFree(halo_Sh);
  delete[] Sz_h;
  delete[] Sh_h;

  return 0;
}
