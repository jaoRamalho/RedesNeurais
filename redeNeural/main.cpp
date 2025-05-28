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

    std::vector<std::vector<double>> inputs;
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
        std::vector<double> inputRow(row.begin() + 1, row.end() - 1); // Ignorar o índice e última coluna
        std::vector<double> outputRow(1, row.back());                 // Última coluna como saída

        inputs.push_back(inputRow);
        outputs.push_back(outputRow);
    }

    file.close();

    // Normalizar os dados de entrada
    inputs = normalizeData(inputs);

    NeuralNetwork nn{{6, 8, 4}}; // Definindo a arquitetura da rede neural
    nn.train(inputs, outputs); // Treinar a rede neural com os dados normalizados
 
    std::cout << "Treinamento concluído!" << std::endl;


    return 0;
}