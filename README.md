# GradFlow: A Custom Automatic Differentiation Library and Neural Network Framework

<img src="GradFlow logo.jpeg" alt="GradFlow Logo" width="100"> 

GradFlow is a Python library that implements automatic differentiation from scratch. It provides the core building blocks for constructing and training neural networks.

## **Key Features:**

* **Automatic Differentiation:** 
    * Implements forward and backward pass automatic differentiation using a custom `GradNode` class.
    * Supports common arithmetic operations, activation functions (ReLU, sigmoid, tanh, etc.), and custom operations.
* **Neural Network Framework:**
    * Provides classes for building neural network components:
        * `Neuron`: Represents a single neuron with weights and biases.
        * `Layer`: Represents a layer of neurons.
        * `MLP`: Represents a Multi-Layer Perceptron (MLP) network.
    * Enables the construction and training of various neural network architectures.
* **Computational Graph Visualization:** 
    * Includes functionality to visualize the computational graph of the neural network, aiding in understanding the flow of data and gradients.

## **Usage:**

1. **Define the Computational Graph:**
   * Create `GradNode` objects to represent variables, constants, and operations.
   * Connect `GradNode` objects to form the computational graph.

2. **Forward Pass:**
   * Perform the forward pass through the graph by accessing the `data` property of the final node.

3. **Backward Pass:**
   * Compute gradients for all nodes in the graph by calling the `backward` method of the final node.

4. **Update Parameters:**
   * Use the computed gradients to update the weights and biases of your neural network using an optimization algorithm (e.g., gradient descent).

* **Magic Methods:** These are special methods in Python (also known as dunder methods) that helps us customize the behavior of built-in operators. 
    * For example, `__add__()` is called when the `+` operator is used with an object. 
    * In `gradFlow`, magic methods are used to overload operators like `+`, `-`, `*`, `/`, etc., for `GradNode` objects, ensuring that the forward and backward passes are correctly handled for these operations.

## **Naming Conventions:**

* **Single Leading Underscore (_):** Indicates that a parameter or variable is intended for internal use only and should not be accessed directly from outside the class or module.
* **Trailing Underscore (var_):** Used to avoid conflicts with Python keywords.

## **Concept of Backprop:**

* **Forward Pass:** In the forward pass, the initial values of all nodes are set, and the output of the graph is computed.
* **Backward Pass:** 
    * Starts from the final node of the graph. 
    * The gradient of the final node with respect to itself is initialized to 1.
    * At each node, the gradient is calculated using the chain rule. 
        * For example, if node `e` is the result of `c + b`, then `c.grad()` is calculated as `e.grad() * 1.0`.
        * If node `e` is the result of `c * b`, then `c.grad()` is calculated as `e.grad() * b`.
        * So, essentially for a child.gradient() = parent.gradient() * $\frac{d}{dchild}parent$.
    * The `_backward()` function within each `GradNode` stores the logic to compute the gradients of its children.

## Installation

```bash
# Clone the repository
git clone https://github.com/anantmehta33/GradFlow.git

# Install dependencies (assuming you have pip installed)
pip install -r requirements.txt
```

## **Contributing:**

We welcome contributions from the community. Here's how you can contribute:

1. **Fork the repository:** Create your own copy of the project on GitHub.
2. **Make changes:** Implement new features, fix bugs, or improve existing code.
3. **Create a pull request:** Submit a pull request to the main repository with your changes.

### **Contributors:**

* Ajay Jagannath
* Anant Mehta

### **Thanks:**

The work of Andrej Karpathy inspired this project.

## **License:**

MIT License
