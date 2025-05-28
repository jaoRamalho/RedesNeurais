# Makefile for Neural Network project

CXX = g++
CXXFLAGS = -std=c++11 -Wall -I.

SRC = main.cpp
OBJ = main.o

all: neural_network

neural_network: $(OBJ)
	$(CXX) $(CXXFLAGS) -o neural_network $(OBJ)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) neural_network

.PHONY: all clean