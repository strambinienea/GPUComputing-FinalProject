# Compiler to use
CC = g++
# Compiler flags
CFLAGS = -std=c++14 -g
INCLUDE = -Iinclude

BUILD_DIR := build
LIB_DIR := lib
TARGET_DIR := bin

# Name of your main cpp file
MAIN = Main.cpp
# Name of the executable
EXECUTABLE = $(TARGET_DIR)/GPUComp_FP
# Name of the library
MMIO_OBJ = $(BUILD_DIR)/mmio.o

all: $(EXECUTABLE)

debug: CFLAGS += -DDEBUG
debug: all

$(EXECUTABLE): src/${MAIN} $(MMIO_OBJ)
	@mkdir -p $(@D)
	$(CC) $^ -o $@ $(INCLUDE) $(CFLAGS)


$(MMIO_OBJ): $(LIB_DIR)/mmio.cpp
	@mkdir -p $(@D)
	$(CC) -c -o $@ $(INCLUDE) $^ $(CFLAGS)

$(LIB_DIR)/mmio.cpp: include/mmio.h

clean:
	rm -rf $(TARGET_DIR)/*
	rm -rf $(BUILD_DIR)/*
