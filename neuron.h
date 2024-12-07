#ifndef NEUTHREAD_NEURON_H
#define NEUTHREAD_NEURON_H

#include <functional>
#include <mutex>
#include <stdexcept>


#include <vector>

class Neuron {
private:
    char activationType;
    std::vector<double> weights;
    double bias;

    double sigmoid(double x);
    double relu(double x);
    double tanhfunc(double x);

public:
    Neuron(char activationType, size_t numInputs);

    double activate(double input);
    double computeOutput(const std::vector<double> &inputs) const;

    void updateWeights(const std::vector<double> &inputs, double learningRate, double delta);

    // Logging function
    void logWeightsAndBias() const;

    // Getters
    char getActivationType() const;

    // Setters
    void setActivationType(char activationType);
};



#endif //NEUTHREAD_NEURON_H
