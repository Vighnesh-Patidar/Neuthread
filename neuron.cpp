#include "Neuron.h"
#include <cmath>
#include <random>
#include <stdexcept>

double Neuron::sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double Neuron::relu(double x) { return x > 0 ? x : 0; }
double Neuron::tanhfunc(double x) { return std::tanh(x); }

Neuron::Neuron(char activation_f, size_t numInputs)
        : activationType(activation_f), weights(numInputs), bias(0.0) {
    if (activation_f != 's' && activation_f != 'r' && activation_f != 't') {
        throw std::invalid_argument("Unsupported activation function");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (auto &weight : weights) {
        weight = dist(gen);
    }
    bias = dist(gen);
}

double Neuron::activate(double input) {
    switch (activationType) {
        case 's': return sigmoid(input);
        case 'r': return relu(input);
        case 't': return tanhfunc(input);
        default: throw std::logic_error("Invalid activation type");
    }
}

double Neuron::computeOutput(const std::vector<double> &inputs) const {
    if (inputs.size() != weights.size()) {
        throw std::invalid_argument("Input size does not match weight size");
    }

    double weightedSum = bias;
    for (size_t i = 0; i < inputs.size(); ++i) {
        weightedSum += inputs[i] * weights[i];
    }

    return activate(weightedSum);
}

void Neuron::updateWeights(const std::vector<double> &inputs, double learningRate, double delta) {
    if (inputs.size() != weights.size()) {
        throw std::invalid_argument("Input size does not match weight size");
    }

    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] -= learningRate * delta * inputs[i];
    }
    bias -= learningRate * delta;
}

void Neuron::logWeightsAndBias() const {
    std::cout << "Weights: ";
    for (const auto &weight : weights) {
        std::cout << weight << " ";
    }
    std::cout << "\nBias: " << bias << "\n";
}
