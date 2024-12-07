#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "Network.h"

void readData(const std::string &filePath, std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &outputs) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filePath);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream stream(line);
        std::vector<double> inputRow;
        std::vector<double> outputRow;
        bool isInput = true;

        double value;
        while (stream >> value) {
            if (isInput) {
                inputRow.push_back(value);
            } else {
                outputRow.push_back(value);
            }
            if (stream.peek() == ',') {
                stream.ignore();
                isInput = false;
            }
        }

        inputs.push_back(inputRow);
        outputs.push_back(outputRow);
    }

    file.close();
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <data file> <epochs>\n";
        return 1;
    }

    std::string dataFilePath = argv[1];
    int epochs = std::stoi(argv[2]);

    // Define network architecture
    std::vector<size_t> layersConfig = {3, 5, 2};  // Example: 3-input, 5-hidden, 2-output
    char activationFunc = 's'; // Sigmoid activation for all layers
    size_t numThreads = 2;     // Number of threads for parallelism

    Network net(layersConfig, activationFunc, numThreads);

    // Read data
    std::vector<std::vector<double>> inputs, expectedOutputs;
    try {
        readData(dataFilePath, inputs, expectedOutputs);
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    // Check input/output consistency
    if (inputs.size() != expectedOutputs.size()) {
        std::cerr << "Input and output data sizes do not match\n";
        return 1;
    }

    // Train the network
    double learningRate = 0.01;
    net.train(inputs, expectedOutputs, learningRate, epochs);

    // Display final weights and biases
    std::cout << "\nFinal Network Weights and Biases:\n";
    for (size_t i = 0; i < layersConfig.size(); ++i) {
        std::cout << "Layer " << i + 1 << ":\n";
        Network::logLayerWeightsAndBiases(i);  // Assumes Network has this function
    }

    return 0;
}
