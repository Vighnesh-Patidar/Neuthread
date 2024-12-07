//
// Created by wishu on 12/4/2024.
//
#include "neuron.h"
#include "Threader.h"
#include <vector>

#ifndef NEUTHREAD_LAYER_H
#define NEUTHREAD_LAYER_H

class Layer {
private:
    std::vector<Neuron> neurons;
    Threader threader;
    std::vector<double> outputs; // Store outputs of the layer for backpropagation

public:
    Layer(char activationFunc, size_t numNeurons, size_t numInputs, size_t numThreads);

    void computeLayer(const std::vector<double> &inputs, std::vector<double> &outputs);
    void backpropagate(const std::vector<double> &inputs,
                       const std::vector<double> &deltas,
                       std::vector<double> &prevDeltas,
                       double learningRate);
    void setLayerActivation(char activationFunc);
    const std::vector<double> &getOutputs() const;
    const std::vector<Neuron> &getNeurons() const; // Add this method

};
#endif //NEUTHREAD_LAYER_H
