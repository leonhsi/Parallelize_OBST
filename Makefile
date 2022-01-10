NVCC = nvcc
CXX=g++
CXXFLAGS=-Iinc/ -Iobjs/ -O3 -std=c++17 -Wall -g -fPIC -lm

APP_NAME=obst
OBJDIR=objs
INCLUDE=inc

CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' 
CUDA_COMPILE_FLAGS = -rdc=true --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3 -ldevcudart


default: $(APP_NAME)

.PHONY: dirs clean

dirs:
		/bin/mkdir -p $(OBJDIR)/

clean:
		/bin/rm -rf $(OBJDIR) *.ppm *~ $(APP_NAME)

OBJS=$(OBJDIR)/main.o $(OBJDIR)/serial.o $(OBJDIR)/layer1.o $(OBJDIR)/layer2.o $(OBJDIR)/dp.o 

$(APP_NAME): dirs $(OBJS)
		$(NVCC) ${CUDA_LINK_FLAGS} -o $@ $(OBJS) 
$(OBJDIR)/%.o: %.cc
		$(CXX) $< $(CXXFLAGS) -c -o $@
$(OBJDIR)/main.o: $(INCLUDE)/CycleTimer.h $(INCLUDE)/serial.h $(INCLUDE)/layer1.h $(INCLUDE)/layer2.h $(INCLUDE)/dp.h
$(OBJDIR)/layer1.o : layer1.cu $(INCLUDE)/layer1.h
	${NVCC} ${CUDA_COMPILE_FLAGS} -c layer1.cu -o $@
$(OBJDIR)/layer2.o : layer2.cu $(INCLUDE)/layer2.h
	${NVCC} ${CUDA_COMPILE_FLAGS} -c layer2.cu -o $@
$(OBJDIR)/dp.o : dp.cu $(INCLUDE)/dp.h
	${NVCC} ${CUDA_COMPILE_FLAGS} -c dp.cu -o $@
