# Micrograd Implementation

A tiny Autograd engine that implements backpropagation over a dynamically built DAG (Directed Acyclic Graph). This is a pure Python implementation of a scalar-valued autograd system, similar to the one taught by Andrej Karpathy.

## Features
- **Scalar Autograd**: Supports basic arithmetic operations (`+`, `-`, `*`, `/`, `**`) and non-linearities (`tanh`, `ReLU`).
- **Dynamic Computation Graph**: Automatically builds the graph during the forward pass.
- **Topological Sort**: Ensures gradients are calculated in the correct order during the backward pass.
- **Neural Network Library**: Includes basic abstractions for `Neuron`, `Layer`, and `MLP` (Multi-Layer Perceptron).

## Mathematical Intuition
Each `Value` object maintains a `_backward` function. When `backward()` is called on the final loss:
1. It performs a **topological sort** to order all nodes from output to input.
2. It applies the **Chain Rule** recursively:
   $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$$
   where $L$ is the loss, $y$ is the output of the current operation, and $x$ is the input.

## Project Structure
- `micrograd.ipynb`: The main notebook containing experiments and visualizations.
- `engine.py`: The core `Value` class and Autograd logic.
- `nn.py`: Neural network components (Neurons, Layers, MLP).

## Verification
The gradients produced by this engine have been verified against **PyTorch** to ensure mathematical correctness.