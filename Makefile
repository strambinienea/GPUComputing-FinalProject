# Compiler to use
CC = nvcc
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

all: $(EXECUTABLE)

debug: CFLAGS += -DDEBUG
debug: all

$(EXECUTABLE): src/${MAIN}
	@mkdir -p $(@D)
	$(CC) $^ -o $@ $(INCLUDE) $(CFLAGS)

clean:
	rm -rf $(TARGET_DIR)/*
	rm -rf $(BUILD_DIR)/*
