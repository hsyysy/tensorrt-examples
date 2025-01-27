CC = g++

CFLAGS = -O3
CFLAGS += -Wall
CFLAGS += `pkg-config --cflags cudart-12.6`
CFLAGS += `pkg-config --libs cudart-12.6`
CFLAGS += -lnvinfer

main:main.cpp
	${CC} -o $@ main.cpp ${CFLAGS}
