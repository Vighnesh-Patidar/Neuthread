//
// Created by wishu on 12/6/2024.
//

#ifndef NEUTHREAD_NETWORK_H
#define NEUTHREAD_NETWORK_H


#include "Layer.h"

class Network {
private:
    std::vector<Layer> layers;

public:
    Network(const std::vector<size_t> &neuronsPerLayer, char activationFunc, size_t numThreads);

    void forward(const std::vector<double> &inputs, std::vector<double> &outputs);
    void train(const std::vector<std::vector<double>> &inputs,
               const std::vector<std::vector<double>> &expectedOutputs,
               double learningRate, int epochs);

    void addLayer(size_t numNeurons, char activationFunc, size_t numThreads);

    // Logging function for layer weights
    static void logLayerWeightsAndBiases(size_t layerIndex);
};


#endif //NEUTHREAD_NETWORK_H
