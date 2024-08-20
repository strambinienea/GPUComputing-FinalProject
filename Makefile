# Compiler to use
CC = nvcc
# Compiler flags
CFLAGS = -std=c++14 -g
INCLUDE = -Iinclude

BUILD_DIR := build
LIB_DIR := lib
TARGET_DIR := bin

# Name of your main cpp file
MAIN = Main.cu
# Name of the executable
EXECUTABLE = $(TARGET_DIR)/GPUComp_FP
# Name of the library
OBJECTS = $(BUILD_DIR)/mmio.o

all: $(EXECUTABLE)

debug: CFLAGS += -DDEBUG
debug: all

$(EXECUTABLE): src/${MAIN} $(OBJECTS)
	@mkdir -p $(@D)
	$(CC) $^ -o $@ $(INCLUDE) $(CFLAGS)

$(OBJECTS): $(LIB_DIR)/mmio.cpp
	@mkdir -p $(@D)
	$(CC) -c -o $@ $(INCLUDE) $^ $(CFLAGS)

$(LIB_DIR)/MatrixLib.cpp: include/mmio.h

clean:
	rm -rf $(TARGET_DIR)/*
	rm -rf $(BUILD_DIR)/*
