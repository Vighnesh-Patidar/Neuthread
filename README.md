# **Neural Network with Multithreading**

This project implements a multithreaded feedforward neural network in C++. The neural network is designed to support efficient forward and backward propagation using multiple threads. The implementation includes customizable layers, neurons, and activation functions.

---

## **Project Structure**

### **Neuron**
Represents a single neuron with weights, a bias, and an activation function.

#### **Key Methods**
- **`Neuron(char activationType, size_t numInputs)`**  
  Initializes weights and bias randomly based on the number of inputs.

- **`double activate(double input)`**  
  Applies the activation function (Sigmoid, ReLU, or Tanh) to a scalar input.

- **`double computeOutput(const std::vector<double> &inputs)`**  
  Calculates the weighted sum of inputs plus bias and applies the activation function.

- **`void updateWeights(const std::vector<double> &inputs, double learningRate, double delta)`**  
  Adjusts the weights and bias based on backpropagation deltas.

- **`void logWeightsAndBias()`**  
  Logs the weights and bias to the console for debugging.

---

### **Layer**
Represents a layer of neurons and manages parallel computation for forward and backward propagation.

#### **Key Methods**
- **`Layer(char activationFunc, size_t numNeurons, size_t numInputs, size_t numThreads)`**  
  Constructor to initialize a layer with a specified activation function, number of neurons, inputs, and threads.

- **`void computeLayer(const std::vector<double> &inputs, std::vector<double> &outputs)`**  
  Performs forward propagation for the layer using multithreading.

- **`void backpropagate(const std::vector<double> &inputs, const std::vector<double> &deltas, std::vector<double> &prevDeltas, double learningRate)`**  
  Performs backpropagation to adjust neuron weights and calculate deltas for the previous layer.

- **`void setLayerActivation(char activationFunc)`**  
  Sets the activation function for all neurons in the layer.

- **`const std::vector<double> &getOutputs() const`**  
  Returns the outputs of the layer.

- **`const std::vector<Neuron> &getNeurons() const`**  
  Returns the neurons in the layer for inspection or logging.

---

### **Network**
Combines multiple layers to form a complete neural network.

#### **Key Methods**
- **`Network(const std::vector<size_t> &neuronsPerLayer, char activationFunc, size_t numThreads)`**  
  Constructor that initializes the network with a specified architecture, activation function, and number of threads.

- **`void addLayer(size_t numNeurons, char activationFunc, size_t numThreads)`**  
  Dynamically adds a layer to the network.

- **`void forward(const std::vector<double> &inputs, std::vector<double> &outputs)`**  
  Performs forward propagation through all layers.

- **`void train(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &expectedOutputs, double learningRate, int epochs)`**  
  Trains the network using backpropagation and gradient descent.

- **`void logLayerWeightsAndBiases(size_t layerIndex)`**  
  Logs weights and biases for all neurons in a specified layer in JSON-like format.

---

### **Threader**
Handles multithreaded computation for layer operations.

#### **Key Methods**
- **`Threader(size_t avThreads, size_t numThreads)`**  
  Constructor that initializes the threader with available and requested threads.

- **`size_t parallelizeLayer(const std::vector<Neuron> &neurons, std::vector<double> &inputs, std::vector<double> &outputs, size_t startIdx, size_t endIdx) const`**  
  Splits neuron computations across multiple threads.

- **`static void setLayerActivation(std::vector<Neuron> &neurons, char activationFunc)`**  
  Sets the activation function for a batch of neurons.

---

## **How to Compile**

Use the following command to compile the project:
```bash
g++ main.cpp Network.cpp Layer.cpp Neuron.cpp Threader.cpp -o Neuthread -pthread
