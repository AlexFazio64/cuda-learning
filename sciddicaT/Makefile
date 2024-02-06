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

EXEC = sciddicaT_tiled_no-halo

default:$(EXEC)

$(EXEC): sciddicaT_tiled_no-halo.cu
	$(CPPC) $^ -o $@ -O3

run:
	./$(EXEC) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) && md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)

clean:
	rm -f $(EXEC) *.o *output*

wipe:
	rm -f *.o *output*
