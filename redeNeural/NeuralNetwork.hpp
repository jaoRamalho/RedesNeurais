#ifndef NeuralNetwork_hpp
#define NeuralNetwork_hpp

#include "Neuron.hpp"
#include "Function.hpp"
#include <utility> 
#include <functional>
#include <string>
#include <unordered_map>

const double LEARNING_RATE = 0.0015; // Learning rate for weight updates
const double EPOCHS = 1500; // Number of training epochs

class NeuralNetwork {
private:
    std::vector<std::vector<Neuron>> layers; // Layers of neurons

public:
    NeuralNetwork(std::vector<int> hiddenLayerSizes) {
        if (hiddenLayerSizes.empty()) {
            throw std::invalid_argument("Hidden layer sizes must not be empty.");
        }

        // Initialize hidden layers
        for (size_t i = 0; i < hiddenLayerSizes.size(); ++i) {
            std::vector<Neuron> layer;
            int numInputs = (i == 0) ? hiddenLayerSizes[i] : hiddenLayerSizes[i - 1];
            for (int j = 0; j < hiddenLayerSizes[i]; ++j) {
                layer.push_back(Neuron(numInputs));
            }
            layers.push_back(layer);
            std::cout << "Creating layer " << i + 1 << " with " << hiddenLayerSizes[i] << " neurons." << std::endl;
        }
    }

    std::vector<double> feedForward(const std::vector<double>& inputs) {
        if (layers.empty()) {
            throw std::invalid_argument("Neural network has no layers.");
        }

        std::vector<double> currentInputs = inputs;

        // Forward pass through each layer
        for (size_t j = 0; j < layers.size() - 1; j++) {
            auto& layer = layers[j];
            std::vector<double> outputs;
            
            for (size_t i = 0; i < layer.size(); i++) {
                Neuron& neuron = layer[i];
                double rawOutput = neuron.activate(currentInputs);
                double sigmoidOutput = sigmoid(rawOutput); // Apply activation function
                neuron.setOutput(sigmoidOutput); // Store activated output in the neuron
                outputs.push_back(sigmoidOutput); // Store raw output before activation
            }
            currentInputs = outputs; // Use outputs as inputs for the next layer
        }

        // Process the last layer with softmax
        std::vector<double> rawOutputs;
        for (size_t i = 0; i < layers.back().size(); ++i) {
            Neuron& neuron = layers.back()[i];
            double rawOutput = neuron.activate(currentInputs);
            rawOutputs.push_back(rawOutput);
        }

        rawOutputs = softmax(rawOutputs);

        for (size_t i = 0; i < rawOutputs.size(); ++i) {
            layers.back()[i].setOutput(rawOutputs[i]); // Store raw output in the last layer neurons
        }

        // Apply softmax to the last layer outputs
    

        return rawOutputs;
    }


    std::vector<int> extract_confusion_stats(const std::vector<std::vector<int>>& matrix) {
        int numClasses = matrix.size();
        
        int totalTP = 0.0;
        int totalTN = 0.0;
        int totalFP = 0.0;
        int totalFN = 0.0;

        for (int k = 0; k < numClasses; ++k) {
            totalTP += matrix[k][k];

            
            for (int i = 0; i < numClasses; ++i) {
                if (i != k) totalFP += matrix[i][k];
            }

            for (int j = 0; j < numClasses; ++j) {
                if (j != k) totalFN += matrix[k][j];
            }

            for (int i = 0; i < numClasses; ++i) {
                for (int j = 0; j < numClasses; ++j) {
                    if (i != k && j != k) totalTN += matrix[i][j];
                }
            }
        }

        return {totalTP, totalTN, totalFP, totalFN};
    }

    std::pair<double, std::vector<int>> calculate_MSE_MC(  const std::vector<std::vector<double>>& expectedOutputs,
                                                                        const std::vector<std::vector<double>>& actualOutputs 
    ) {
        if (expectedOutputs.size() != actualOutputs.size()) {
            throw std::invalid_argument("Expected and actual outputs must have the same size.");
        }

        int numClasses = 4;
        std::vector<std::vector<int>> confusionMatrix(numClasses, std::vector<int>(numClasses, 0));
        double mse = 0.0;

        for (size_t i = 0; i < expectedOutputs.size(); ++i) {

            // MSE para todas as saídas da amostra
            for (int j = 0; j < numClasses; ++j) {
                double error = expectedOutputs[i][j] - actualOutputs[i][j];
                mse += error * error;
            }

            int expectedClass = std::distance(expectedOutputs[i].begin(),
                                            std::max_element(expectedOutputs[i].begin(), expectedOutputs[i].end()));

            int predictedClass = std::distance(actualOutputs[i].begin(),
                                            std::max_element(actualOutputs[i].begin(), actualOutputs[i].end()));

            confusionMatrix[expectedClass][predictedClass]++;
        }

        mse /= (expectedOutputs.size() * numClasses); // MSE médio por saída

        return {mse, extract_confusion_stats(confusionMatrix)};
    }


    void backPropagate(const std::vector<double>& expectedOutputs) {
       //calculate oara a camada de saida
      // std::cout << "Backpropagating through the network." << std::endl;

        if (layers.empty()) {
            throw std::invalid_argument("Neural network has no layers.");
        }

        for(size_t i = 0; i < layers.back().size(); ++i) {
            Neuron& neuron = layers.back()[i];
            double output = neuron.getOutput();
            double error = expectedOutputs[i] - output;
            //std::cout << "Expected Output: " << expectedOutputs[i] << ", Actual Output: " << output << std::endl;

            double delta = error;
            neuron.setDelta(delta);
        }

        // Start from the last layer and move backwards
        for (int j = layers.size() - 2; j >= 0; j--) {
            auto& layer = layers[j];
            for (size_t i = 0; i < layer.size(); ++i) {
                Neuron& neuron = layer[i];

                double output = neuron.getOutput();
                
                double error = expectedOutputs[i] - output;

                double delta = error * sigmoidDerivative(output); // Derivative of sigmoid
                neuron.setDelta(delta);
            }
        }

        for(int j = layers.size() - 1; j >= 0; j--){
            auto& layer = layers[j];
            // Update weights and biases for the current layer
            for (size_t i = 0; i < layer.size(); ++i) {
                Neuron& neuron = layer[i];
                std::vector<double>& weights = neuron.getWeights();
                for (size_t k = 0; k < weights.size(); ++k) {
                    weights[k] += LEARNING_RATE * neuron.getDelta() *  neuron.getInputs()[k]; // Update weights
                }
                // Update bias
                double newBias = neuron.getBias() + LEARNING_RATE * neuron.getDelta();
                neuron.setBias(newBias);
            }
        }
    }


    void train(const std::vector<std::vector<double>>& trainingData,
               const std::vector<std::vector<double>>& expectedOutputs) {
        if (trainingData.empty() || expectedOutputs.empty() || trainingData.size() != expectedOutputs.size()) {
            throw std::invalid_argument("Training data and expected outputs must be non-empty and of the same size.");
        }

        // Abrir arquivo para salvar os resultados do MSE
        
        int sizeLayers = layers.size();
        std::vector<int> sizesNeurons;
        for (int i = 0; i < sizeLayers; ++i) {
            sizesNeurons.push_back(layers[i].size());
        }

        std::string fileName = "data/results_";
        //decreva camadas
        fileName += "(";
        for (size_t i = 0; i < sizesNeurons.size(); ++i) {
            fileName += "-" + std::to_string(sizesNeurons[i]);
        }
        fileName += ")_";
        //decreva epocas
        fileName += std::to_string(EPOCHS);
        fileName += "_";
        //decreva taxa de aprendizado
        fileName += std::to_string(LEARNING_RATE); // Multiplying by 1000 for better readability
        fileName += ".csv";

        std::ofstream resultFile(fileName);
        if (!resultFile.is_open()) {
            throw std::runtime_error("Failed to open file for writing MSE results.");
        }

        resultFile << "Epoch,MSE,RMSE,Accuracy,Precision,Recall,F1Score\n"; // Header for the CSV file
        
        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            double totalMSE = 0.0;
            double totalTP = 0.0;
            double totalTN = 0.0;
            double totalFP = 0.0;
            double totalFN = 0.0;


            for (size_t i = 0; i < trainingData.size(); ++i) {
                const auto& inputs = trainingData[i];
                const auto& expectedOutput = expectedOutputs[i];
                std::vector<double> expectedOutputCompare = {0.0, 0.0, 0.0, 0.0};
                expectedOutputCompare[expectedOutput[0] - 1] = 1.0; // Convert expected output to one-hot encoding
                
                // Forward pass
                std::vector<double> actualOutput = feedForward(inputs);

                // Calculate MSE
                std::pair<double, std::vector<int>> result = calculate_MSE_MC({expectedOutputCompare}, {actualOutput});
                double mse = result.first;
                std::vector<int> confusionStats = result.second;

                totalMSE += mse; // Accumulate MSE for the epoch
                totalTP += confusionStats[0]; // True Positives
                totalTN += confusionStats[1]; // True Negatives
                totalFP += confusionStats[2]; // False Positives
                totalFN += confusionStats[3]; // False Negatives

                //std::cout << totalMSE << " - " << totalTP << " - " << totalTN << " - " << totalFP << " - " << totalFN << std::endl;
            

                // Backward pass
                backPropagate(expectedOutputCompare);
            }        
            totalMSE /= trainingData.size(); // Average MSE for the epoch
            double totalRMSE = std::sqrt(totalMSE); // Root Mean Square Error
            
            double totalAccuracy = (totalTP + totalTN) / (totalTP + totalTN + totalFP + totalFN);
            
            double totalPrecision = (totalTP + totalFP > 0) ? totalTP / (totalTP + totalFP) : 0.0; // Precision
            
            double totalRecall = (totalTP + totalFN > 0) ? totalTP / (totalTP + totalFN) : 0.0; // Recall
            
            double totalF1Score = (totalPrecision + totalRecall > 0) ? 2 * (totalPrecision * totalRecall) / (totalPrecision + totalRecall) : 0.0; // F1 Score

            resultFile  << epoch + 1 << "," 
                        << totalMSE << "," 
                        << totalRMSE << "," 
                        << totalAccuracy << "," 
                        << totalPrecision << "," 
                        << totalRecall << "," 
                        << totalF1Score << "\n";
        }            
        resultFile.close();
        std::cout << "Training completed. Results saved to " << fileName << std::endl;
    }

    std::vector<double> predict(const std::vector<double>& inputs) {
        return feedForward(inputs);
    }
};

#endif // NeuralNetwork_hpp