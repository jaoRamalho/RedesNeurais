#ifndef FUNCTION_HPP
#define FUNCTION_HPP


#include <cmath>
#include <algorithm>
#include <vector>

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x)); // Sigmoid activation function
}

double sigmoidDerivative(double x) {
    return std::exp(-x) / std::pow(1.0 + std::exp(-x), 2); // Derivative of sigmoid
}

double customTanh(double x) {
    return std::tanh(x); // Hyperbolic tangent activation function
}

double customTanhDerivative(double x) {
    return 1.0 - std::pow(std::tanh(x), 2); // Derivative of tanh
}

std::vector<double> softmax(const std::vector<double>& inputs) {
    double maxInput = *std::max_element(inputs.begin(), inputs.end());
    double sumExp = 0.0;

    std::vector<double> probabilities(inputs.size());
    for (double input : inputs) {
        sumExp += std::exp(input - maxInput); // Subtract max for numerical stability
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        probabilities[i] = std::exp(inputs[i] - maxInput) / sumExp; // Softmax for each input
    }

    return probabilities;
}

double relu(double x) {
    return std::max(0.0, x); // ReLU activation function
}

double reluDerivative(double x) {
    return (x > 0) ? 1.0 : 0.0; // Derivative of ReLU
}


#endif // FUNCTION_HPP