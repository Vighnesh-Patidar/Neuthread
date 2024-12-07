//
// Created by wishu on 12/3/2024.
// This implementation parallelizes neural network layer calculations using multiple threads.
//

#ifndef NEUTHREAD_THREADER_H
#define NEUTHREAD_THREADER_H

#include <vector>
#include <thread>
#include "neuron.h"

class Threader {
private:
    size_t avThreads;
    size_t numThreads;

public:
    Threader(size_t avThreads, size_t numThreads);
    size_t parallelizeLayer(const std::vector<Neuron> &neurons,
                            std::vector<double> &inputs,
                            std::vector<double> &outputs,
                            size_t startIdx,
                            size_t endIdx) const;
    static void setLayerActivation(std::vector<Neuron> &neurons, char activation_f);
};

#endif // NEUTHREAD_THREADER_H
