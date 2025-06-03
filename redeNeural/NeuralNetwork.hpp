#ifndef NeuralNetwork_hpp
#define NeuralNetwork_hpp

#include "Neuron.hpp"
#include "Function.hpp"
#include <utility>
#include <functional>
#include <string>
#include <unordered_map>

extern double LEARNING_RATE; // Learning rate for weight updates
const double EPOCHS = 10000; // Number of training epochs
extern std::vector<std::vector<int>> MATRIZ_CONFUSAO;

class NeuralNetwork {
private:
    std::vector<std::vector<Neuron>> layers; // Layers of neurons

public:
    NeuralNetwork(std::vector<int> hiddenLayerSizes, std::vector<ActivationFunction> activationFunctions, int numInputs);

    std::vector<double> feedForward(const std::vector<double>& inputs);

    std::vector<int> extract_confusion_stats(const std::vector<std::vector<int>>& matrix);

    std::pair<double, std::vector<int>> calculate_MSE_MC(const std::vector<std::vector<double>>& expectedOutputs, const std::vector<std::vector<double>>& actualOutputs);

    void backPropagate(const std::vector<double>& expectedOutputs, double posExpected);

    void trainClassification(const std::vector<std::vector<double>>& trainingData,
                             const std::vector<std::vector<double>>& expectedOutputs);

    void trainRegression(const std::vector<std::vector<double>>& trainingData, const std::vector<double>& expectedOutputs);
    
    std::vector<double> predict(const std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& expectedOutputs);

    double returnActivationFunction(ActivationFunction activationFunctionName, double input);
    double returnDerivateActivationFunction(ActivationFunction activationFunctionName, double input);
};

#endif // NeuralNetwork_hpp