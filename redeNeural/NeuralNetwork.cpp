#include "NeuralNetwork.hpp"
#include <fstream>
#include <filesystem> 
#include <sstream>  


std::vector<std::vector<int>> MATRIZ_CONFUSAO = {
    {0, 0, 0, 0}, // Classe 1
    {0, 0, 0, 0}, // Classe 2
    {0, 0, 0, 0}, // Classe 3
    {0, 0, 0, 0}  // Classe 4
};

double LEARNING_RATE = 0.02; // Global learning rate for weight updates

NeuralNetwork::NeuralNetwork(std::vector<int> hiddenLayerSizes, std::vector<ActivationFunction> activationFunctions, int numInputs) {
    if (hiddenLayerSizes.empty()) {
        throw std::invalid_argument("Hidden layer sizes must not be empty.");
    }

    for (size_t i = 0; i < hiddenLayerSizes.size(); ++i) {
        std::vector<Neuron> layer;

        for (int j = 0; j < hiddenLayerSizes[i]; ++j) {
            layer.push_back(Neuron((i == 0)? numInputs : hiddenLayerSizes[i - 1], activationFunctions[i]));
        }
        layers.push_back(layer);
        std::cout << "Creating layer " << i + 1 << " with " << hiddenLayerSizes[i] << " neurons." << std::endl;
    }

}

std::vector<double> NeuralNetwork::feedForward(const std::vector<double>& inputs) {
    if (layers.empty()) {
        throw std::invalid_argument("Neural network has no layers.");
    }

    std::vector<double> currentInputs = inputs;

    for (size_t j = 0; j < layers.size() - 1; j++) {
        auto& layer = layers[j];
        std::vector<double> outputs;

        for (size_t i = 0; i < layer.size(); i++) {
            Neuron& neuron = layer[i];
            double rawOutput = neuron.activate(currentInputs);
            double funcOutput = returnActivationFunction(neuron.getActivationFunction(), rawOutput);
            neuron.setOutput(funcOutput);
            outputs.push_back(funcOutput);
        }
        currentInputs = outputs;
    }

    std::vector<double> rawOutputs;
    if (layers.back()[0].getActivationFunction() == ActivationFunction::Softmax) {
        for (size_t i = 0; i < layers.back().size(); ++i) {
            Neuron& neuron = layers.back()[i];
            double rawOutput = neuron.activate(currentInputs);
            rawOutputs.push_back(rawOutput);
        }
        
        rawOutputs = softmax(rawOutputs);
        
        for (size_t i = 0; i < rawOutputs.size(); ++i) {
            layers.back()[i].setOutput(rawOutputs[i]);
        }
    } else {
        for (size_t i = 0; i < layers.back().size(); ++i) {
            Neuron& neuron = layers.back()[i];
            double rawOutput = neuron.activate(currentInputs);
            double funcOutput = returnActivationFunction(neuron.getActivationFunction(), rawOutput);
            rawOutputs.push_back(funcOutput);
            neuron.setOutput(funcOutput); // Store the raw output
        }
    }

    return rawOutputs;
}

std::vector<int> NeuralNetwork::extract_confusion_stats(const std::vector<std::vector<int>>& matrix) {
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

std::pair<double, std::vector<int>> NeuralNetwork::calculate_MSE_MC(const std::vector<std::vector<double>>& expectedOutputs, const std::vector<std::vector<double>>& actualOutputs) {
    if (expectedOutputs.size() != actualOutputs.size()) {
        throw std::invalid_argument("Expected and actual outputs must have the same size.");
    }

    int numClasses = 4;
    std::vector<std::vector<int>> confusionMatrix(numClasses, std::vector<int>(numClasses, 0));
    double mse = 0.0;

    for (size_t i = 0; i < expectedOutputs.size(); ++i) {
        for (int j = 0; j < numClasses; ++j) {
            double error = expectedOutputs[i][j] - actualOutputs[i][j];
            mse += error * error;
        }

        int expectedClass = std::distance(expectedOutputs[i].begin(), std::max_element(expectedOutputs[i].begin(), expectedOutputs[i].end()));
        int predictedClass = std::distance(actualOutputs[i].begin(), std::max_element(actualOutputs[i].begin(), actualOutputs[i].end()));

        confusionMatrix[expectedClass][predictedClass]++;
        MATRIZ_CONFUSAO [expectedClass][predictedClass]++;
    }


    mse /= (expectedOutputs.size() * numClasses);

    return {mse, extract_confusion_stats(confusionMatrix)};
}

void NeuralNetwork::backPropagate(const std::vector<double>& expectedOutputs,  double posExpected) {
    if (layers.empty()) {
        throw std::invalid_argument("Neural network has no layers.");
    }

    std::vector<double> weigthsClass = {6.15, 2, 3.84, 50}; // Weights for the softmax layer

    int sub = (layers.back()[0].getActivationFunction() == ActivationFunction::Softmax) ? 2 : 1;

    if (sub == 2) {
        for (size_t i = 0; i < layers.back().size(); ++i) {
            Neuron& neuron = layers.back()[i];
            double output = neuron.getOutput();
            double error = expectedOutputs[i] - output;
            // For softmax, we use the raw output as the delta
            double delta = error * ((posExpected == i) ? 1.0 : weigthsClass[i]);
            neuron.setDelta(delta);
        }
    }

    for (int j = layers.size() - sub; j >= 0; j--) {
        auto& layer = layers[j];
        for (size_t i = 0; i < layer.size(); ++i) {
            Neuron& neuron = layer[i];

            double output = neuron.getOutput();
            double error = expectedOutputs[i] - output;

            double delta = error * returnDerivateActivationFunction(neuron.getActivationFunction(), output);
            neuron.setDelta(delta);
        }
    }

    for (int j = layers.size() - 1; j >= 0; j--) {
        auto& layer = layers[j];
        for (size_t i = 0; i < layer.size(); ++i) {
            Neuron& neuron = layer[i];
            std::vector<double>& weights = neuron.getWeights();
            // Atualização de peso com descida de gradiente e L2 regularização
            for (size_t k = 0; k < weights.size(); ++k) {
                double grad = neuron.getDelta() * neuron.getInputs()[k];
                weights[k] += LEARNING_RATE * grad;
            }
            // Atualização de bias
            double newBias = neuron.getBias() + LEARNING_RATE * neuron.getDelta();
            neuron.setBias(newBias);
        }
    }
}


std::string generateUniqueFileName(const std::vector<int>& sizesNeurons, const std::vector<ActivationFunction>& activationFunctions, double learningRate, int epochs, std::string baseName = "data/training/stats_") {

    std::string salvaBase = baseName;

    // Adicionar número de camadas e neurônios
    baseName += "(";
    for (size_t i = 0; i < sizesNeurons.size(); ++i) {
        baseName += "-" + std::to_string(sizesNeurons[i]);
    }
    baseName += ")_";

    // Adicionar funções de ativação
    baseName += "(";
    for (size_t i = 0; i < activationFunctions.size(); ++i) {
        baseName += "-" + std::to_string(static_cast<int>(activationFunctions[i])); // Converte enum para int
    }
    baseName += ")_";

    // Adicionar taxa de aprendizado e épocas
    baseName += std::to_string(static_cast<int>(learningRate * 100000)); // Remove o ponto decimal
    baseName += "_";
    baseName += std::to_string(epochs);

    // Verificar se o arquivo já existe e adicionar numeração
    std::string fileName = baseName + ".csv";

    int counter = 1;
    while (std::filesystem::exists(fileName)) {
        if (salvaBase == "data/results/result_") {
            return fileName; 
        }

        fileName = baseName + "_" + std::to_string(counter) + ".csv";
        counter++;
    }

    if (salvaBase == "data/results/result_") {
        std::ofstream resultFile(fileName);
        resultFile << "Accuracy\n"; // Cria o arquivo com o cabeçalho
        resultFile.close();
    }

    return fileName;
}

void NeuralNetwork::trainClassification(const std::vector<std::vector<double>>& trainingData, const std::vector<std::vector<double>>& expectedOutputs) {
    if (trainingData.empty() || expectedOutputs.empty() || trainingData.size() != expectedOutputs.size()) {
        throw std::invalid_argument("Training data and expected outputs must be non-empty and of the same size.");
    }

    int layerCount = layers.size();
    std::vector<int> sizesNeurons;
    std::vector<ActivationFunction> activationFunctions;
    for (int i = 0; i < layerCount; ++i) {
        sizesNeurons.push_back(layers[i].size());
        activationFunctions.push_back(layers[i][0].getActivationFunction());
    }

    std::string fileName = generateUniqueFileName(sizesNeurons, activationFunctions, LEARNING_RATE, EPOCHS);

    std::ofstream resultFile(fileName);
    if (!resultFile.is_open()) {
        throw std::runtime_error("Failed to open file for writing MSE results.");
    }

    resultFile << "Epoch,MSE,RMSE,Accuracy,Precision,Recall,F1Score\n";

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
            expectedOutputCompare[expectedOutput[0] - 1] = 1.0;

            std::vector<double> actualOutput = feedForward(inputs);

            // //exibe exepected e actual
            // std::cout << "Expected: ";
            // for (const auto& val : expectedOutputCompare) {
            //     std::cout << val << " ";
            // }
            // std::cout << "\nActual: ";
            // for (const auto& val : actualOutput) {
            //     std::cout << val << " ";
            // }
            // std::cout << std::endl;

            std::pair<double, std::vector<int>> result = calculate_MSE_MC({expectedOutputCompare}, {actualOutput});
            double mse = result.first;
            std::vector<int> confusionStats = result.second;

            totalMSE += mse;
            totalTP += confusionStats[0];
            totalTN += confusionStats[1];
            totalFP += confusionStats[2];
            totalFN += confusionStats[3];

            backPropagate(expectedOutputCompare, expectedOutput[0] - 1);
        }
        totalMSE /= trainingData.size();
        double totalRMSE = std::sqrt(totalMSE);

        double totalAccuracy = (totalTP + totalTN) / (totalTP + totalTN + totalFP + totalFN);

        double totalPrecision = (totalTP + totalFP > 0) ? totalTP / (totalTP + totalFP) : 0.0;

        double totalRecall = (totalTP + totalFN > 0) ? totalTP / (totalTP + totalFN) : 0.0;

        double totalF1Score = (totalPrecision + totalRecall > 0) ? 2 * (totalPrecision * totalRecall) / (totalPrecision + totalRecall) : 0.0;

        resultFile << epoch + 1 << "," << totalMSE << "," << totalRMSE << "," << totalAccuracy << "," << totalPrecision << "," << totalRecall << "," << totalF1Score << "\n";
    }
    resultFile.close();
    std::cout << "Training completed. Results saved to " << fileName << std::endl;
}

void NeuralNetwork::trainRegression(const std::vector<std::vector<double>> &trainingData, const std::vector<double> &expectedOutputs){
    
}

std::vector<double> NeuralNetwork::predict(const std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& expectedOutputs) {
    std::vector<double> predictions;

    double sucessCount = 0.0;
    int index = 0;
    for (const auto& input : inputs) {
        std::vector<double> output = feedForward(input);
        double resposta = std::distance(output.begin(), std::max_element(output.begin(), output.end())) + 1; // Assuming the output is a class index starting from 1
        predictions.push_back(resposta); // Assuming we want the first output for predictions        

        if (resposta == expectedOutputs[index][0]) {
            sucessCount++;
        }
        index++;
    }


    int layersCount = layers.size();
    std::vector<int> sizesNeurons;
    std::vector<ActivationFunction> activationFunctions;
    for (int i = 0; i < layersCount; ++i) {
        sizesNeurons.push_back(layers[i].size());
        activationFunctions.push_back(layers[i][0].getActivationFunction());
    }
    std::string fileName = generateUniqueFileName(sizesNeurons, activationFunctions, LEARNING_RATE, EPOCHS, "data/results/result_");
    std::ofstream resultFile(fileName, std::ios::app);
    if (!resultFile.is_open()) {
        throw std::runtime_error("Failed to open file for writing prediction results.");
    }
    resultFile << sucessCount / inputs.size() << "\n";
    resultFile.close();
    
    return predictions;
}

double NeuralNetwork::returnActivationFunction(ActivationFunction activationFunctionName, double input) {
    switch (activationFunctionName){
    case ActivationFunction::Sigmoid:
        return sigmoid(input); 
    case ActivationFunction::Tanh:
        return customTanh(input); 
    case ActivationFunction::ReLU:
        return relu(input); 
    case ActivationFunction::derivativeSigmoid:
        return sigmoid(input);
    case ActivationFunction::derivativeTanh:
        return customTanh(input);
    case ActivationFunction::derivativeReLU:
        return relu(input);
    case ActivationFunction::none:
        return input; // For the 'none' case, we return the input as is
    default:
        throw std::invalid_argument("Unknown activation function");
    }
    return 0.0; // Default return value if no case matches
}


double NeuralNetwork::returnDerivateActivationFunction(ActivationFunction activationFunctionName, double input) {
    switch (activationFunctionName){
    case ActivationFunction::Sigmoid:
        return sigmoidDerivative(input); 
    case ActivationFunction::Tanh:
        return customTanhDerivative(input); 
    case ActivationFunction::ReLU:
        return reluDerivative(input);
    case ActivationFunction::derivativeSigmoid:
        return sigmoidDerivative(input);
    case ActivationFunction::derivativeTanh:
        return customTanhDerivative(input);
    case ActivationFunction::derivativeReLU:
        return reluDerivative(input);
    case ActivationFunction::none:
        return 1.0; // 
    default:
        throw std::invalid_argument("Unknown activation function"); 
    }
    return 0.0; // Default return value if no case matches
}
