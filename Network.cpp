#include "Network.h"
#include <stdexcept>
#include <iostream>

Network::Network(const std::vector<size_t> &neuronsPerLayer, char activationFunc, size_t numThreads) {
    size_t numInputs = neuronsPerLayer[0]; // Input size for the first layer
    for (size_t i = 1; i < neuronsPerLayer.size(); ++i) {
        addLayer(neuronsPerLayer[i], activationFunc, numThreads);
        numInputs = neuronsPerLayer[i];
    }
}

void Network::addLayer(size_t numNeurons, char activationFunc, size_t numThreads) {
    size_t numInputs = layers.empty() ? 0 : layers.back().getOutputs().size();
    layers.emplace_back(activationFunc, numNeurons, numInputs, numThreads);
}

void Network::forward(const std::vector<double> &inputs, std::vector<double> &outputs) {
    std::vector<double> layerInputs = inputs;
    for (auto &layer : layers) {
        layer.computeLayer(layerInputs, outputs);
        layerInputs = outputs; // Output of this layer is input to the next
    }
}

void Network::train(const std::vector<std::vector<double>> &inputs,
                    const std::vector<std::vector<double>> &expectedOutputs,
                    double learningRate, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalError = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            std::vector<double> outputs;
            forward(inputs[i], outputs);

            // Compute error
            std::vector<double> deltas(outputs.size());
            for (size_t j = 0; j < outputs.size(); ++j) {
                double error = outputs[j] - expectedOutputs[i][j];
                deltas[j] = error; // Store delta for backpropagation
                totalError += 0.5 * error * error; // Mean Squared Error
            }

            // Backpropagate
            std::vector<double> prevDeltas = deltas;
            for (size_t layerIdx = layers.size(); layerIdx-- > 0;) {
                std::vector<double> newDeltas;
                layers[layerIdx].backpropagate(layerIdx == 0 ? inputs[i] : layers[layerIdx - 1].getOutputs(),
                                               prevDeltas,
                                               newDeltas,
                                               learningRate);
                prevDeltas = newDeltas;
            }
        }

        std::cout << "Epoch " << epoch + 1 << "/" << epochs
                  << " - Total Error: " << totalError << "\n";
    }
}

void Network::logLayerWeightsAndBiases(size_t layerIndex) {
    if (layerIndex >= layers.size()) {
        throw std::out_of_range("Invalid layer index");
    }

    std::cout << "Logging weights and biases for layer " << layerIndex + 1 << ":\n";
    for (const auto &neuron : layers[layerIndex].getNeurons()) {
        neuron.logWeightsAndBias();
    }
}
