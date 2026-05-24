CC = clang++
CFLAGS = -Wall -Wextra -std=c++17
LDFLAGS = -framework Metal -framework MetalKit -framework Cocoa -framework QuartzCore

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

TARGET = $(BIN_DIR)/liquifier
SOURCES = $(wildcard $(SRC_DIR)/*.mm)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.mm=$(OBJ_DIR)/%.o)

SHADERS_SRC = $(SRC_DIR)/Shaders.metal
SHADERS_LIB = $(BIN_DIR)/default.metallib

all: $(TARGET) $(SHADERS_LIB)

$(TARGET): $(OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.mm
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(SHADERS_LIB): $(SHADERS_SRC)
	@mkdir -p $(BIN_DIR)
	xcrun -sdk macosx metal -c $< -o $(OBJ_DIR)/Shaders.air
	xcrun -sdk macosx metallib $(OBJ_DIR)/Shaders.air -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean
