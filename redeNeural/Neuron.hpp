#ifndef NEURON_HPP
#define NEURON_HPP


#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>


class Neuron {
private:
    std::vector<double> weights;
    std::vector<double> inputs;
    double bias;
    double output;
    double delta;

public:
    Neuron(int numInputs=0) {
        // Initialize weights and bias with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        weights.resize(numInputs);
        for (double& weight : weights) {
            weight = dis(gen); // Random weight initialization
        }
        bias = dis(gen); // Random bias initialization
        output = 0.0;
        delta = 0.0; // Initialize delta to zero
    }

    Neuron() {
        // Default constructor initializes with no inputs
        weights = {};
        bias = 0.0;
        output = 0.0;
        delta = 0.0; // Initialize delta to zero
    
    }

    double activate(const std::vector<double>& inputs) {
        if (inputs.size() != weights.size()) {
            throw std::invalid_argument("Input size does not match weight size.");
        }

        this->inputs = inputs; // Store inputs for later use

        output = bias; // Start with the bias
        for (size_t i = 0; i < inputs.size(); ++i) {
            output += inputs[i] * weights[i]; // Weighted sum
        }
        return output; // Return the raw output (before activation)
    }

    double getOutput() const {
        return output;
    }

    void setOutput(double outputValue) {
        output = outputValue;
    }

    void setDelta(double deltaValue) {
        delta = deltaValue;
    }

    double getDelta() const {
        return delta;
    }

    std::vector<double>& getWeights() {
        return weights;
    }

    double getBias() const {
        return bias;
    }

    void setBias(double biasValue) {
        bias = biasValue;
    }

    std::vector<double>& getInputs() {
        return inputs; // Store inputs for later use
    }

};

#endif // NEURON_HPP