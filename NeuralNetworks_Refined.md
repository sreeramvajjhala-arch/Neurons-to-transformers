# Introduction to Neural Networks & Generative AI

## Core Concepts: AI, ML, and Deep Learning

- **AI (Artificial Intelligence):** Any machine capable of reasoning, solving problems, and thinking like a human. AI is the broad field that encompasses Machine Learning (ML).
- **ML (Machine Learning):** A primary method to achieve AI.
- **Deep Learning:** A subset of ML that uses layered artificial neural networks to model complex patterns in data.

### Why Deep Learning?
Deep Learning is particularly powerful when:
- Dealing with **large datasets**.
- Analyzing and predicting **complex, non-linear patterns**.
- **High computational power** is available.

It effectively **mimics the human brain** using interconnected **artificial neurons**.

---

## Neural Network Fundamentals

### The Universal Approximation Theorem
This foundational theorem states that a neural network with a sufficient number of layers can **approximate any continuous function**. This is the core premise of deep learning.

### Categories of Machine Learning
1.  **Supervised Learning:** Uses labeled data.
2.  **Unsupervised Learning:** Finds patterns in unlabeled data.
3.  **Reinforcement Learning:** Learns through feedback from actions (e.g., DeepSeek's training).

---

## Steps to Build a Neural Network

1.  **Define the Problem Statement** (can be supervised, unsupervised, or reinforced).
2.  **Collect and Prepare Data**.
3.  **Split Data** (e.g., 70% Train, 15% Validate, 15% Test).
4.  **Define the Neural Network Architecture**.
5.  **Train the Neural Network**.
6.  **Validate the Neural Network**.
7.  **Test the Neural Network**.
8.  **Deploy the Neural Network**.
9.  **Monitor** performance.

### Key Term: Epoch
An **epoch** is one full pass of the entire training dataset through the learning algorithm. Models are often trained for multiple epochs (e.g., 10-20).

---

## The Mathematics Behind Neural Networks

### The Goal: Minimizing Loss
The objective of training is to minimize the difference between the model's prediction and the actual value from the dataset. This difference is called the **loss** or **cost**.

A common loss function is the **Mean Squared Error (MSE)**:
$$ C = (y_{true} - y_{predicted})^2 $$

### Simple Linear Example
Let's model a straight-line relationship:
$$ y = wx + b $$
Where:
- $ y $ is the predicted output
- $ x $ is the input feature
- $ w $ is the **weight** (parameter)
- $ b $ is the **bias** (parameter)
- $ z $ is the true value from the dataset

The cost function becomes:
$$ C = (z - y)^2 $$

To minimize $ C $, we adjust $ w $ and $ b $ using **Gradient Descent**.

#### Gradient Descent Update Rules
$$ w_{new} = w_{old} - \alpha \frac{\partial C}{\partial w} $$
$$ b_{new} = b_{old} - \alpha \frac{\partial C}{\partial b} $$
Where $ \alpha $ is the **learning rate**.

#### Calculating the Gradients (Using the Chain Rule)
$$ \frac{\partial C}{\partial w} = \frac{\partial C}{\partial y} \cdot \frac{\partial y}{\partial w} $$
$$ \frac{\partial C}{\partial y} = -2(z - y) $$
$$ \frac{\partial y}{\partial w} = x $$
Therefore:
$$ \frac{\partial C}{\partial w} = -2x(z - y) $$

Similarly, for the bias:
$$ \frac{\partial y}{\partial b} = 1 $$
$$ \frac{\partial C}{\partial b} = -2(z - y) $$

Intuitively, the model starts with random guesses for $ w $ and $ b $ and iteratively updates them to reduce the cost until a minimum is found.

### Multi-Feature Example: House Price Prediction
Problem: Predict house price ($ y $) based on area ($ x_1 $) and location ($ x_2 $).

The model function becomes:
$$ y = w_1x_1 + w_2x_2 + b $$
The training process involves finding the optimal values for $ w_1 $, $ w_2 $, and $ b $ that minimize the cost $ C = (z - y)^2 $ by computing $ \frac{\partial C}{\partial w_1} $, $ \frac{\partial C}{\partial w_2} $, and $ \frac{\partial C}{\partial b} $.

### What are Weights and Bias?
- **Weight ($ w $):** Represents the importance or influence of a specific input feature on the output.
- **Bias ($ b $):** Allows the model to fit the data better by shifting the activation function away from the origin, providing flexibility.

These **weights** and **biases** are collectively known as the model's **parameters**.

---

## The Training Process

The standard training loop for a neural network, including LLMs, is as follows:

1.  **Initialize** weights and biases with random values.
2.  **Forward Pass:** Input data is passed through the network to compute the predicted output.
3.  **Calculate Loss:** Compute the cost (e.g., MSE) between the prediction and the true value.
4.  **Backward Propagation:** Calculate the gradients of the loss with respect to all parameters ($ \frac{\partial C}{\partial w} $, $ \frac{\partial C}{\partial b} $) using the chain rule.
5.  **Update Parameters:** Adjust the weights and biases using the gradients and the learning rate.
6.  **Repeat** steps 2-5 for multiple epochs until the loss is minimized.

---

## Large Language Models (LLMs)

LLMs are a buzzword in the ML/AI industry, built on complex neural network architectures like the **Transformer**.

### Examples of LLMs:
- **Claude** (Anthropic)
- **ChatGPT** (OpenAI)
- **Gemini** (Google)
- **DeepSeek** (DeepSeek AI)
- **Mistral** (Mistral AI)

Similarly, there are also **Small Language Models (SLMs)**.

**GPT** stands for **Generative Pre-trained Transformer**.

### Path to Understanding & Building
To effectively build AI agents and apps, it's crucial to understand the underlying architectures:
1.  Start with fundamental concepts of **ANN (Artificial Neural Networks)**.
2.  Progress to **RNN (Recurrent Neural Networks)** and **LSTM (Long Short-Term Memory)** networks for sequential data.
3.  Finally, study the **Transformer Architecture**, which is the foundation of modern LLMs.

---

## Next Topic: Non-Linearity
*[This section was cut off in the original notes and should be expanded next.]*

---