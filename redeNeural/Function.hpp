#ifndef FUNCTION_HPP
#define FUNCTION_HPP


#include <cmath>
#include <algorithm>
#include <vector>


enum class ActivationFunction {
    Sigmoid,
    Tanh,
    Softmax,
    ReLU,
    derivativeSigmoid,
    derivativeTanh,
    derivativeReLU,
    none
};


inline double crossEntropy(const std::vector<double>& expected, const std::vector<double>& actual) {
    double loss = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] > 0) { // Avoid log(0)
            loss -= expected[i] * std::log(actual[i]);
        }
    }
    return loss; // Cross-entropy loss
}


inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x)); // Sigmoid activation function
}

inline double sigmoidDerivative(double x) {
    return std::exp(-x) / std::pow(1.0 + std::exp(-x), 2); // Derivative of sigmoid
}

inline double customTanh(double x) {
    return std::tanh(x); // Hyperbolic tangent activation function
}

inline double customTanhDerivative(double x) {
    return 1.0 - std::pow(std::tanh(x), 2); // Derivative of tanh
}

inline std::vector<double> softmax(const std::vector<double>& inputs) {
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

inline double relu(double x) {
    return std::max(0.0, x); // ReLU activation function
}

inline double reluDerivative(double x) {
    return (x > 0) ? 1.0 : 0.0; // Derivative of ReLU
}


#endif // FUNCTION_HPP