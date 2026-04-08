# 🧠 Perceptron

A clean, minimal implementation of the Perceptron learning algorithm from scratch using NumPy.

---

## What is a Perceptron?

The Perceptron is one of the simplest and most fundamental neural network architectures. It's a linear binary classifier that learns to separate data through iterative weight updates. Think of it as a single neuron that can learn to make binary decisions based on input features.

### How it Works

The Perceptron operates in two phases:

**1. Forward Pass (Prediction)**
```
ŷ = step(w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ + b)
```

Where:
- `w` = weights learned during training
- `x` = input features
- `b` = bias term (allows shifting the decision boundary)
- `step()` = activation function (returns 1 or -1)

**2. Learning Phase (Weight Update)**
```
Δw = learning_rate × (y_true - ŷ) × x
w = w + Δw
```

The algorithm adjusts weights only when predictions are wrong, gradually converging toward an optimal solution.

---


## Installation

```bash
# Using uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

**Requirements:**
- Python ≥ 3.12
- NumPy ≥ 2.4.4
- Pandas ≥ 3.0.2
- scikit-learn (for utilities)

---

## Usage

### Basic Example

```python
from perceptron import Perceptron
import numpy as np

# Create and train the perceptron
model = Perceptron(learning_rate=0.1, n_iters=10)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | 0.01 | Controls how much weights change per iteration (0-1) |
| `n_iters` | int | 1000 | Number of training iterations |

### Full Example

Run the included example on the Iris dataset:

```bash
python main.py
```

This will:
1. Load the Iris dataset
2. Split into train/test sets (75%/25%)
3. Train the Perceptron to classify Setosa vs. other species
4. Evaluate accuracy on the test set

---

## Project Structure

```
Perceptron/
├── perceptron.py      # Core Perceptron class implementation
├── main.py            # Example usage with Iris dataset
├── pyproject.toml     # Project configuration
└── README.md          # This file
```

---

## Key Concepts

### The Bias Term
The bias acts as an independent offset that allows the decision boundary to shift without being tied to the origin. It's essential for learning non-trivial decision boundaries. [Learn more →](https://stackoverflow.com/questions/2480650/what-is-the-role-of-the-bias-in-neural-networks)

### Learning Rate
The learning rate determines the step size during weight updates. Too high and the model oscillates wildly; too low and training stalls. Finding the sweet spot is crucial. [Explore this concept →](https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/)

### The Step Function
The Perceptron uses a step function as its activation, outputting either 1 or -1. This makes it a linear classifier capable of solving linearly separable problems.

---

## Learning Resources

This implementation is built on foundational concepts from:

1. **[Understanding the Perceptron](https://medium.com/p/15716806ef64)** — Intuitive explanation of how perceptrons learn and adapt  

2. **[Perceptron Learning Algorithm Details](https://medium.com/p/4e5c9300f79f)** — Deeper dive into the mathematics behind weight updates

3. **[The Role of Bias in Neural Networks](https://stackoverflow.com/questions/2480650/what-is-the-role-of-the-bias-in-neural-networks)** — Why bias is crucial for neural network flexibility

4. **[Understanding the Learning Rate](https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/)** — How learning rate affects convergence and model performance

---

## Limitations

The Perceptron algorithm has important constraints:

- ⚠️ **Only works on linearly separable data** — Can't solve XOR or other non-linear problems
- ⚠️ **Binary classification only** — Outputs just 1 or -1
- ⚠️ **No probability estimates** — Returns hard predictions, not confidence scores

For more complex problems, consider Multi-Layer Perceptrons (MLPs) or other neural network architectures.

---

## Visualizing Learning

The implementation tracks errors across iterations. Lower error over time indicates successful learning—a sign that the algorithm is converging toward good weight values.

```
Training progress (errors per iteration):
[45, 42, 38, 35, 32, 28, 25, 22, 18, 15, ...]  ← Error decreases ✓
```

---

## Mathematical Foundation

The Perceptron learning rule is elegantly simple:

```
For each training sample (x, y):
    ŷ = predict(x)
    If ŷ ≠ y:
        w ← w + learning_rate · (y - ŷ) · x
        b ← b + learning_rate · (y - ŷ)
```

This rule drives the convergence guarantee: **if data is linearly separable, the Perceptron will find a solution in finite time.**

---

## Contributing

This is an educational implementation. Feel free to experiment with:
- Different learning rates and iteration counts
- Alternative datasets
- Different activation functions
- Weight initialization strategies

---

## License

Open source and free to use and modify.
