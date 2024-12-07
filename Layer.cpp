#include "Layer.h"
#include <stdexcept>

Layer::Layer(char activationFunc, size_t numNeurons, size_t numInputs, size_t numThreads)
        : threader(std::thread::hardware_concurrency(), numThreads) {
    if (numNeurons == 0) {
        throw std::invalid_argument("Layer must have at least one neuron");
    }
    neurons.reserve(numNeurons);
    for (size_t i = 0; i < numNeurons; ++i) {
        neurons.emplace_back(activationFunc, numInputs);
    }
}

void Layer::computeLayer(const std::vector<double> &inputs, std::vector<double> &outputs) {
    if (inputs.empty()) {
        throw std::invalid_argument("Inputs cannot be empty");
    }

    outputs.resize(neurons.size());
    threader.parallelizeLayer(neurons, const_cast<std::vector<double> &>(inputs), outputs, 0, neurons.size());
    this->outputs = outputs; // Store computed outputs
}

void Layer::backpropagate(const std::vector<double> &inputs,
                          const std::vector<double> &deltas,
                          std::vector<double> &prevDeltas,
                          double learningRate) {
    prevDeltas.resize(inputs.size(), 0.0);

    double clipValue = 5.0;  // Gradient clipping threshold

    for (size_t i = 0; i < neurons.size(); ++i) {
        double clippedDelta = std::max(-clipValue, std::min(clipValue, deltas[i]));
        neurons[i].updateWeights(inputs, learningRate, clippedDelta);

        // Compute deltas for the previous layer
        for (size_t j = 0; j < inputs.size(); ++j) {
            prevDeltas[j] += clippedDelta * neurons[i].activate(inputs[j]);
        }
    }
}

void Layer::setLayerActivation(char activationFunc) {
    for (auto &neuron : neurons) {
        neuron.setActivationType(activationFunc);
    }
}

const std::vector<double> &Layer::getOutputs() const {
    return outputs;
}
const std::vector<Neuron> &getNeurons() const { return neurons; }

