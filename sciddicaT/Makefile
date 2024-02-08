ifndef CPPC
	CPPC=nvcc
endif

# STRESS_TEST_R = 1

ifdef STRESS_TEST_R
	HDR=./data/stress_test_R_header.txt
	DEM=./data/stress_test_R_dem.txt
	SRC=./data/stress_test_R_source.txt
	OUT=./stress_test_R_output
	STEPS=4000
else
	HDR=./data/tessina_header.txt
	DEM=./data/tessina_dem.txt
	SRC=./data/tessina_source.txt
	OUT=./tessina_output
	STEPS=4000
endif

EXEC = sciddicaT_unified
EXEC = sciddicaT_standard
EXEC = sciddicaT_tiled_no-halo
EXEC = sciddicaT_tiled_halo
EXEC = sciddicaT_tiled_halo-larger

GRID_1 = 1
ifdef GRID_1
	G1 = 8
	G2 = 8
endif

# GRID_2 = 1
ifdef GRID_2
	G1 = 16
	G2 = 32
endif

# GRID_3 = 1
ifdef GRID_3
	G1 = 32
	G2 = 16
endif

# GRID_4 = 1
ifdef GRID_4
	G1 = 32
	G2 = 32
endif

default:$(EXEC)

$(EXEC): $(EXEC).cu
	$(CPPC) $^ -O3

run:
	./a.out $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) $(G1) $(G2) && md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)

profile:
	nvprof --metrics flops_sp ./a.out $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) $(G1) $(G2)

clean:
	rm -f $(EXEC) *.o *output*

wipe:
	rm -f *.o a.out *output*
