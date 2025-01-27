CC = g++

CUDA_VER = 12.6

CFLAGS = -O3
CFLAGS += -Wall
CFLAGS += `pkg-config --cflags cudart-${CUDA_VER}`
CFLAGS += `pkg-config --libs cudart-${CUDA_VER}`
CFLAGS += -lnvinfer

main:main.cpp
	${CC} -o $@ main.cpp ${CFLAGS}
