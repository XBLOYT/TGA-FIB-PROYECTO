CUDA_HOME   = /Soft/cuda/12.2.2
CUDA_HOME   = /Soft/cuda/12.2.2

NVCC        = $(CUDA_HOME)/bin/nvcc
ARCH        = -gencode arch=compute_86,code=sm_86
NVCC_FLAGS  = -w -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include $(ARCH) --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc  
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
EXE01           = filtrarBITS.exe
EXE02           = filtrarBW.exe
EXE03           = filtrarColorSplit.exe

EXE04           = filtrarBITSP.exe
EXE05           = filtrarBWP.exe
EXE06           = filtrarColorSplitP.exe

OBJ01           = filtrarBITS.o
OBJ02           = filtrarBW.o
OBJ03           = filtrarColorSplit.o
OBJ04           = filtrarBITSP.o
OBJ05           = filtrarBWP1kernel.o
OBJ06           = filtrarColorSplitPthreadbypixel.o
OBJ07            = filtrarColorSplitPthreadbychannel.o
OBJ08            = filtrarColorSplitPmultikernel.o
OBJ10           = filtrarBWP2kernels.o

default: filtrarBWP2kernels filtrarColorSplitPthreadbypixel filtrarBITSP

seq: filtrarBW filtrarColorSplit filtrarBITS
	
all: default seq

filtrarBITS.o: filtrarBITS.cu
	$(NVCC) -c -o $@ filtrarBITS.cu  $(NVCC_FLAGS)  -I/Soft/stb/20200430  

filtrarBITS: $(OBJ01)
	$(NVCC) $(OBJ01) -o $(EXE01) $(LD_FLAGS)

filtrarBW.o: filtrarBW.cu
	$(NVCC) -c -o $@ filtrarBW.cu  $(NVCC_FLAGS)  -I/Soft/stb/20200430  

filtrarBW: $(OBJ02)
	$(NVCC) $(OBJ02) -o $(EXE02) $(LD_FLAGS)

filtrarColorSplit.o: filtrarColorSplit.cu
	$(NVCC) -c -o $@ filtrarColorSplit.cu  $(NVCC_FLAGS)  -I/Soft/stb/20200430  

filtrarColorSplit: $(OBJ03)
	$(NVCC) $(OBJ03) -o $(EXE03) $(LD_FLAGS)

filtrarBITSP.o: filtrarBITSP.cu
	$(NVCC) -c -o $@ filtrarBITSP.cu  $(NVCC_FLAGS)  -I/Soft/stb/20200430  

filtrarBITSP: $(OBJ04)
	$(NVCC) $(OBJ04) -o $(EXE04) $(LD_FLAGS)

filtrarBWP1kernel.o: filtrarBWP1kernel.cu
	$(NVCC) -c -o $@ filtrarBWP1kernel.cu  $(NVCC_FLAGS)  -I/Soft/stb/20200430  

filtrarBWP1kernel: $(OBJ05)
	$(NVCC) $(OBJ05) -o $(EXE05) $(LD_FLAGS)

filtrarBWP2kernels.o: filtrarBWP2kernels.cu
	$(NVCC) -c -o $@ filtrarBWP2kernels.cu  $(NVCC_FLAGS)  -I/Soft/stb/20200430  

filtrarBWP2kernels: $(OBJ10)
	$(NVCC) $(OBJ10) -o $(EXE05) $(LD_FLAGS)

filtrarColorSplitPthreadbypixel.o: filtrarColorSplitPthreadbypixel.cu
	$(NVCC) -c -o $@ filtrarColorSplitPthreadbypixel.cu  $(NVCC_FLAGS)  -I/Soft/stb/20200430  

filtrarColorSplitPthreadbychannel.o: filtrarColorSplitPthreadbychannel.cu
	$(NVCC) -c -o $@ filtrarColorSplitPthreadbychannel.cu  $(NVCC_FLAGS)  -I/Soft/stb/20200430  

filtrarColorSplitPmultikernel.o : filtrarColorSplitPmultikernel.cu
	$(NVCC) -c -o $@ filtrarColorSplitPmultikernel.cu $(NVCC_FLAGS) -I/Soft/stb/20200430

filtrarColorSplitPthreadbypixel: $(OBJ06)
	$(NVCC) $(OBJ06) -o $(EXE06) $(LD_FLAGS)

filtrarColorSplitPthreadbychannel: $(OBJ07)
	$(NVCC) $(OBJ07) -o $(EXE06) $(LD_FLAGS)

filtrarColorSplitPmultikernel: $(OBJ08)
	$(NVCC) $(OBJ08) -o $(EXE06) $(LD_FLAGS)

clean:
	rm -rf *.o *.exe *output.* FILTRAR.e1* FILTRAR.o1* Out* submit-FILTRO.*
