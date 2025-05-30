# Makefile for Neural Network project

CXX = g++
CXXFLAGS = -std=c++17 -Wall -I.

SRC = main.cpp NeuralNetwork.cpp
OBJ = main.o NeuralNetwork.o

all: clean neural_network

neural_network: $(OBJ)
	$(CXX) $(CXXFLAGS) -o neural_network $(OBJ)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) neural_network

.PHONY: all clean