#include "Threader.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>

// Constructor to initialize Threader with available threads
Threader::Threader(size_t avThreads, size_t numThreads)
        : avThreads(avThreads), numThreads(numThreads) {
    if (numThreads > avThreads) {
        throw std::logic_error("Requested threads exceed available threads");
    }
}

// Parallelize layer computation
size_t Threader::parallelizeLayer(const std::vector<Neuron> &neurons,
                                  std::vector<double> &inputs,
                                  std::vector<double> &outputs,
                                  size_t startIdx,
                                  size_t endIdx) const {
    size_t numElements = endIdx - startIdx;
    if (numElements == 0) {
        throw std::invalid_argument("Invalid range for parallel computation");
    }

    // Ensure outputs vector is large enough
    if (outputs.size() < numElements) {
        outputs.resize(numElements);
    }

    // Divide work among threads
    size_t chunkSize = (numElements + numThreads - 1) / numThreads;
    std::vector<std::thread> threads;

    // Lambda to handle computation for a range of indices
    auto computeRange = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            outputs[i] = neurons[i].activate(inputs[i]);
        }
    };

    // Launch threads
    for (size_t t = 0; t < numThreads; ++t) {
        size_t rangeStart = startIdx + t * chunkSize;
        size_t rangeEnd = std::min(rangeStart + chunkSize, endIdx);

        if (rangeStart < rangeEnd) { // Ensure there's work for this thread
            threads.emplace_back(computeRange, rangeStart, rangeEnd);
        }
    }

    // Join all threads
    for (auto &thread : threads) {
        thread.join();
    }

    return numThreads;
}

// Set the activation function for a layer of neurons
void Threader::setLayerActivation(std::vector<Neuron> &neurons, char activation_f) {
    for (auto &neuron : neurons) {
        neuron.setActivationType(activation_f);
    }
}
