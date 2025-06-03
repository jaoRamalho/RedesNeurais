#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include "NeuralNetwork.hpp"

// Função para normalizar os dados
std::vector<std::vector<double>> normalizeData(const std::vector<std::vector<double>>& data) {
    std::vector<std::vector<double>> normalizedData = data;
    size_t numColumns = data[0].size();
    std::vector<double> minValues(numColumns, std::numeric_limits<double>::max());
    std::vector<double> maxValues(numColumns, std::numeric_limits<double>::lowest());

    // Encontrar os valores mínimos e máximos para cada coluna
    for (const auto& row : data) {
        for (size_t i = 0; i < numColumns; ++i) {
            minValues[i] = std::min(minValues[i], row[i]);
            maxValues[i] = std::max(maxValues[i], row[i]);
        }
    }

    // Normalizar cada valor
    for (auto& row : normalizedData) {
        for (size_t i = 0; i < numColumns; ++i) {
            row[i] = (row[i] - minValues[i]) / (maxValues[i] - minValues[i]);
        }
    }

    return normalizedData;
}

int main() {
    std::ifstream file("../treino_sinais_vitais_com_label.txt");
    if (!file.is_open()) {
        std::cerr << "Erro ao abrir o arquivo!" << std::endl;
        return 1;
    }

    std::vector<std::vector<double>> allInputs;
    std::vector<std::vector<double>> outputs;

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;

        // Parsear os valores da linha
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value));
        }

        // Separar entradas e saídas
        std::vector<double> inputRow(row.begin() + 1, row.end() - 2); // Ignorar o índice e última coluna
        std::vector<double> outputRow(1, row.back());                 // Última coluna como saída

        allInputs.push_back(inputRow);
        outputs.push_back(outputRow);
    }

    file.close();

    //contando a ocorrencia de cada classe
    std::vector<int> classCounts(4, 0); // Para classes 1-4
    for (const auto& output : outputs) {
        int label = static_cast<int>(output[0]);
        if (label >= 1 && label <= 4) {
            classCounts[label - 1]++;
        }
    }
    // Exibir a contagem de cada classe
    std::cout << "Contagem de classes:" << std::endl;
    for (size_t i = 0; i < classCounts.size(); ++i) {
        std::cout << "Classe " << (i + 1) << ": " << classCounts[i] << " ocorrências" << std::endl;
    }
    // Porcentagem de cada classe
    std::cout << "Porcentagem de classes:" << std::endl;
    for (size_t i = 0; i < classCounts.size(); ++i) {
        double percentage = (static_cast<double>(classCounts[i]) / outputs.size()) * 100.0;
        std::cout << "Classe " << (i + 1) << ": " << percentage << "%" << std::endl;
    }

    // Normalizar os dados de entrada
    allInputs = normalizeData(allInputs);

    
    // MELHORES RESULTADOS ATINGIDOS
    // FUNCOES -> TOPOLOGIA -> TAXA DE APRENDIZADO
    //(3, 1, 2) -> (8, 16, 4) -> 0.01
    //(3,7,2) -> (8, 16, 4) -> 0.005
    //(0, 7, 2) -> (16, 8, 4) -> 0.015
    //(3, 7, 2) -> (16, 8, 4) -> 0.012


    //treino recebe metade dos inputs
    std::vector<std::vector<double>> trainingInputs(allInputs.begin(), allInputs.begin() + allInputs.size() / 2);
    std::vector<std::vector<double>> classificationInputs(allInputs.begin() + allInputs.size() / 2, allInputs.end());
    
    std::vector<std::vector<double>> outputsTraning(outputs.begin(), outputs.begin() + outputs.size() / 2);
    std::vector<std::vector<double>> outputsClassification(outputs.begin() + outputs.size() / 2, outputs.end());
    

  
    NeuralNetwork nn{
        {3, 4,4}, 
        {ActivationFunction::Tanh, ActivationFunction::none, ActivationFunction::Softmax},
        5
    }; // Definindo a arquitetura da rede neural

    nn.trainClassification(trainingInputs, outputsTraning); // Treinar a rede neural com os dados normalizados
 
    std::cout << "Treinamento concluído!" << std::endl;

    // Testar a rede neural com os dados de classificaçãonn

    std::vector<double> predictions = nn.predict(allInputs, outputs);
}