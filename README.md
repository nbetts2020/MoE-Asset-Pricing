# Natural Language for Asset Pricing

Traditional models for asset pricing, such as the Capital Asset Pricing Model (CAPM), the Arbitrage Pricing Theory (APT), and the Black-Scholes model, have relied on linear relationships between risk factors and returns, focusing primarily on financial metrics like market risk and volatility. These models, while foundational, often fail to capture the complex, non-linear interactions present in modern financial markets and broader economic factors.

Recent research, such as [Deep Learning for Asset Pricing by Chen et al. (2023)](https://arxiv.org/abs/1904.00745) and [Structural Deep Learning in Conditional Asset Pricing by Fan et al. (2022)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4117882), demonstrates the effectiveness of deep learning in handling the non-linearities and time-varying dynamics of asset prices. These approaches leverage advanced techniques like feed-forward networks, LSTMs, and GANs to incorporate a wide range of economic and firm-specific factors, providing a more flexible and data-driven method for predicting future prices and risk premiums.

Building upon an abundance of research in Large Language Models (LLMs) and Mixture-of-Experts (MoE) architectures, this project introduces a transformer-based SparseMoE Language Model tailored for asset price prediction, wherein the model's final fully connected (dense) layer consists of one output unit (where the output is a singular continuous scalar value), as opposed to a probability distribution. Drawing from extensive textual data related to a stock in the form of articles, news reports, and financial statements, the model predicts the stock price 1 month out.

## Key Features

**Sparse Mixture-of-Experts (MoE) Architecture**: An architecture in which $k$ experts are divvied up to solve a task, has been popular technique as of recent, boasting similar performance yet less compute compared to their monolithic counterparts. To illustrate why this might be useful, let's think about the case of next token prediction. When handing the prompt “Tell me about President Eisenhower’s first inaugural address” to an LLM, little to no activation is present amongst the majority of the model’s neurons. In this instance, the model will not activate neurons tailored for subjects that are irrelevant towards the prompt. To name a few, neurons focused on technical computing, abstract mathematical theories, and deep scientific concepts, among many others, will stay inactive as their expertise holds, at most, a tangential connection with the prompt. This is, quite frankly, extremely inefficient. When dealing with a task that's already computationally expensive and faced with a shortage of GPUs, this inefficiency only compounds the issues. Thus, MoEs attempt to curtail this inefficiency by creating a number of models that each specialize in a certain task.

**16k Context Window**: A relatively large context window was chosen to ensure the model has access to as much relevant information as possible. For a given input, the current article, along with its metadata (title, publication, author, etc.), are included in the context. Additionally, the previous 10 articles related to the stock are provided. Building on this further, stock prices from 96 hours, 48 hours, and 24 hours prior to the article's release, as well as the price at the time of the article's release, are included. This comprehensive data allows the model to consider both historical articles and market trends when making predictions.

**Custom Training Pipeline:** Optimized for training on NVIDIA A100 GPUs, employing techniques like mixed precision and gradient checkpointing for efficient resource utilization.

**FlashAttention 2**: Implemented for efficient and scalable attention computation, enabling the model to handle long sequences effectively.

## Model Architecture

**Total Parameters**: Approximately **254 million parameters**, which includes all shared parameters and parameters from all experts across all transformer blocks.

**Active Parameters**: Approximately **84 million parameters** are active during a forward pass. This includes the shared parameters and the parameters from the experts that are selected by the routing mechanism.

### Model Components

1. **Embedding Layers**:
   - **Token Embedding**: Converts input tokens into embeddings of dimension $n_{\text{embed}} = 192$.
   - **Positional Embedding**: Adds positional information to token embeddings to maintain the sequence order.

2. **Transformer Blocks** (Total of 96 layers):
   - **Multi-Head FlashAttention**:
     - **Heads**: $n_{\text{head}} = 12$ attention heads.
     - **FlashAttention**: Efficient computation of attention mechanisms for long sequences.
     - **Attention Calculation**:

       $\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right) V$

   - **Layer Normalization**: Applied before attention and MoE layers to stabilize training.

   - **Sparse Mixture-of-Experts (MoE) Layer**:
     - **Experts**: 8 experts in total, each a feed-forward network (MLP) with two linear layers and ReLU activation.
     - **Noisy Top-$k$ Routing**:
       - Routes tokens to the top $k = 2$ experts based on a gating mechanism.
       - **Routing Mechanism**:

         $\begin{align*}\text{Noisy Logits} & = W_r x + \epsilon \odot \sigma(W_n x) \\\text{Top-}k & = \text{Indices of top } k \text{ elements in Noisy Logits} \\\text{Router Output} & = \text{Softmax}(\text{Sparse\_Logits})\end{align*}$

       - $W_r$ and $W_n$ are learnable parameters, $\epsilon$ is Gaussian noise, and $\sigma$ is the softplus activation.

     - **Expert Processing**:
       - Each selected expert processes its assigned tokens independently.
       - Outputs are combined and passed on to the next layer.

   - **Residual Connections**: Each sub-layer includes a residual path to facilitate gradient flow.

3. **Output Layers**:
   - **Layer Normalization**: Final normalization before regression.
   - **Mean Pooling**: Averages the sequence embeddings to obtain a fixed-size vector.
   - **Regression Head**: A linear layer that outputs the predicted stock price as a continuous scalar value:

     $\text{Predicted Price} = W_{\text{reg}} x_{\text{pooled}} + b_{\text{reg}}$

### Training Details

- **Loss Function**: Mean Squared Error (MSE) between the predicted and actual stock prices.

  $\text{Loss} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$

- **Optimizer**: AdamW optimizer with a learning rate of $2 \times 10^{-5}$.

- **Batch Size**: 16 sequences per batch.

- **Epochs**: Trained over 20 epochs.

- **Techniques Used**:
  - **Gradient Checkpointing**: Reduces memory usage by recomputing intermediate activations during the backward pass.
  - **Mixed Precision Training**: Utilizes half-precision floating points to speed up training and reduce memory consumption.
  - **FlashAttention 2**: Efficient attention mechanism for handling long sequences.

---

This model architecture leverages advanced neural network components to capture complex dependencies between textual data and future stock prices. By integrating a high-capacity SparseMoE transformer with efficient attention mechanisms, the model effectively processes and learns from vast amounts of unstructured text, enabling more accurate and insightful financial forecasting.
