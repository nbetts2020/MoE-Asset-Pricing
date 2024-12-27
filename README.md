# Natural Language for Asset Pricing

Traditional models for asset pricing, such as the Capital Asset Pricing Model (CAPM), the Arbitrage Pricing Theory (APT), and the Black-Scholes model, have relied on linear relationships between risk factors and returns, focusing primarily on financial metrics like market risk and volatility. These models, while foundational, often fail to capture the complex, non-linear interactions present in modern financial markets and broader economic factors.

Recent research, such as "Deep Learning for Asset Pricing by Chen et al. (2019)"[^1] and "Structural Deep Learning in Conditional Asset Pricing by Fan et al. (2022)"[^2], demonstrate the effectiveness of deep learning in handling the non-linearities and time-varying dynamics of asset prices. These approaches leverage architectures such as feed-forward networks, LSTMs, and GANs to incorporate a wide range of economic and firm-specific factors, providing a more flexible and data-driven method for predicting future prices and risk premiums.

Building upon an abundance of research in Large Language Models (LLMs) and Mixture-of-Experts (MoE) architectures, this project introduces a transformer-based SparseMoE Language Model tailored for asset price prediction for small cap stocks, wherein the model's final fully connected (dense) layer consists of one output unit (where the output is a singular continuous scalar value), as opposed to a probability distribution. Drawing from extensive textual data related to a stock in the form of articles, news reports, and financial statements, the model, along with complementary tools such as Online Learning and an optimal context selection algorithm, predicts the stock price 30 days out.

## Data[^3]

SC454k is a dataset of roughly 454k news articles and press releases related to small cap stocks from nasdaq.com and attached with market data from the Trade and Quote Millisecond dataset (TAQ/MSEC) from Wharton Research Data Services. The schema of the data looks something like this:

- **Symbol**: the ticker symbol of respective stock (e.g., AAPL)
- **Security**: the full name of stock associated with the ticker symbol (e.g., Apple Inc.)
- **Sector**: broader economic sector the stock belongs to, sourced from nasdaq.com (e.g., Technology)
- **Industry**: more specific business segment within 'Sector', sourced from nasdaq.com (e.g., Computer Manufacturing)
- **URL**: URL associated with the 'Article' column
- **Date**: the date the 'Article' was published, in the format of MMM DD, YYYY HH:MM AM/PM ET (e.g., Aug 06, 2024 12:11 PM ET)
- **RelatedStocksList**: stocks/topics that are related or appear in the 'Article' (excluding the respective stock mentioned in the 'Symbol' column). Delineated by '|'s (e.g., Markets|INTC|UNH|DOW)
- **Article**: text of the news article or press release. Any specially formatted text (bold text, tables, etc.) are treated in a pseudo-LaTeX format. For example, bold text are surrounded by '**'s and tables are wrapped in \begin{table}{...}\end{table}
- **Title**: title of the news article or press release
- **articleType**: value denoting the type of 'Article', either 'News' or 'Press Release'
- **Publication**: source responsible for publishing the 'Article' (e.g., The Motley Fool)
- **Author**: person or entity responsible for publishing the 'Article'. May appear as an entity similar to the 'Publication' (e.g., Publication: Zacks, Author: Zacks Equity Research) or as individual person (e.g., Publication: Validea, Author: John Reese)
- **Risk_Free_Rate**: annualized 1-month U.S. Treasury bill yield at the time of the article's publication - rate is sourced from Federal Reserve Economic Data (FRED), where values are expressed as decimals
- **weighted_avg_x_hours**: stock price 'x' hours away from 'Date' (calculated from a weighted average of the 8 closest stock prices to 'Date')
  
**Scraping/Pairing Market Data**

Scraping the non-pricing aspects of this dataset was conducted across 10 EC2 t2.large instances across two parts: scraping links, then the content of these links. Puppeteer, a headless Chrome browser in JavaScript, was used for lightweight, quick scraping. Data was subsequently cleaned with a combination of PySpark and NumPy. Pairing this with market data was made available through Wharton Research Data Services and their extensive collection of financial datasets. One collection of these datasets is the Trade and Quote Millisecond dataset (TAQ/MSEC), consisting of millisecond level access to the tick-by-tick trade and quote data of all activity within the U.S. National Market System. The process of pairing this pricing data with the timestamped news article/press release data through their API was magnitudes more tricky than meets the eye. The issue comes down to one thing: latency. The granularity of the API call can only be day-level, meaning a simple call to get the pricing data for say `Apr. 3, 2020 11:13:37AM` will return around 10-15 thousand records of data, and because multiple prices are being retrieved for each news article/press release (I'm grabbing the price of the stock 4 days before, 2 days before, all the way to 30 days after with more granularity in between, totaling 18 different prices across 10 unique days), this is a large amount of data to pair, even with highly-optimized C++ code. Thus, the pairing algorithm went through a number of iterations to optimize efficiency such as promoting a more indexed-based approach to finding dates, a stringent approach to clearing out unused memory, and customized bash scripts for easy deployment of multiple EC2 instances. Ultimately, totaling around 1400 hours of compute across 41 (yes, 41) m7g.medium instances. Any values where no market data was available is denoted with a null value. The number of rows with complete market data totals just above 440k.

Inspiration for this dataset was taken from "FNSPID: A Comprehensive Financial News Dataset in Time Series"[^4].

## Key Features

**Sparse Mixture-of-Experts (MoE) Architecture**[^5]: An architecture in which $k$ experts are divvied up to solve a task, a popular technique as of recent, boasting similar performance yet less compute compared to their monolithic counterparts. To illustrate why this might be useful, let's think about the case of next token prediction. When handing the prompt “Tell me about President Eisenhower’s first inaugural address” to an LLM, little to no activation is present amongst the majority of the model’s neurons. In this instance, the model will not activate neurons tailored for subjects that are irrelevant towards the prompt. To name a few, neurons focused on technical computing, abstract mathematical theories, and deep scientific concepts, among many others, will stay inactive as their expertise holds, at most, a tangential connection with the prompt. This is, quite frankly, extremely inefficient. When dealing with a task that's already computationally expensive and faced with a shortage of GPUs, this inefficiency only compounds the issues. Thus, MoEs attempt to curtail this inefficiency by creating a number of models that each specialize in a certain task.

**64k Context Window/Optimal Context Selection**[^6][^7][^8]: A large context window was chosen to ensure the model has access to as much relevant information as possible. This is made possible by leveraging an Energy-Based Model and Monte Carlo Sampling to evaluate and score each article-context pair to identify the most pertinent information. The EBM assigns lower energy values to more useful contexts based on their Mean Squared Error (MSE) in predicting stock prices, while Monte Carlo Sampling uses these energy values to probabilistically select higher-scoring contexts. The context sampled from includes the following components:

- **Broader Economic Information**
- **Industry-Specific Information**
- **Sector-Specific Information**
- **Stock-Specific Information** (sampling from articles related to the stock's biggest market movers)
- **Last Nine Articles and Associated Pricing Data** related to the stock (including the current article)

This combination enables the model to focus on the most relevant data within the given sample space, enhancing prediction accuracy.

**Custom Training Pipeline:** Optimized for training for the NVIDIA Ampere architecture, employing techniques like mixed precision and gradient checkpointing for efficient resource utilization.

**FlashAttention 2**[^9][^10]: Implemented for efficient and scalable attention computation, enabling the model to handle long sequences effectively.

**Online Learning**[^11][^12]: Designed to continuously adapt to new data streams, ensuring the model remains up-to-date with the latest market trends and information. By leveraging techniques such as Synaptic Intelligence, Memory Replay Buffers, Elastic Weight Consolidation, and others, the model effectively mitigates catastrophic forgetting, maintaining its predictive accuracy over time while incorporating new insights.

## Model Architecture

**Total Parameters**: Approximately **1.009 billion parameters**, which includes all shared parameters and parameters from all experts across all transformer blocks.

**Active Parameters**: Approximately **328 million parameters** are active during a forward pass. This includes the shared parameters and the parameters from the experts that are selected by the routing mechanism.

## Training Details

- **Loss Function**: MSE between the predicted and actual stock prices.

  $\text{Loss} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$

- **Optimizer**: AdamW optimizer with a base learning rate of $2 \times 10^{-5}$. Layer-wise learning rate decay is applied with a decay factor of $0.95$ per layer, starting from the deepest layers. The embedding layers and regression head use the base learning rate, while deeper layers have progressively smaller learning rates due to the decay.

- **Batch Size**: 16 sequences per batch.

- **Epochs**: Trained for up to 20 epochs, with early stopping enabled if the loss does not improve for 5 consecutive epochs (configurable via `--early_stopping_patience` argument).

- **# of Experts:** 8

- **# of Active Experts per Token:** 2

- **Tokenizer:** GPT-2

- **Techniques Used**:
  - **Gradient Checkpointing**[^13][^14]: Reduces memory usage by recomputing intermediate activations during the backward pass.
  - **Mixed Precision Training**: Utilizes half-precision floating points to speed up training and reduce memory consumption.
  - **FlashAttention 2**: Efficient attention mechanism for handling long sequences.
  - **Layer-wise Learning Rate Decay (LLRD)**[^15]: Ensures more stable updates by applying smaller learning rates to lower layers and higher rates to upper layers, improving convergence.

As financial analysis is defined by changing markets, it only makes sense to pair it with an architecture that caters well to its inherent modality. The disparity amongst inputs makes this a suitable candidate for a MoE, and provides contribution to an area of research that has previously not been explored in-depth.

Inspiration for the basic components of this architecture were taken from terrific work of Avinash Sooriyarachchi - forked from his [MakeMoE](https://github.com/AviSoori1x/makeMoE) repo.

## Energy-Based Model and Monte Carlo Sampling for Optimal Context Selection [^6][^7][^8]

The inspiration for a context optimal algorithm (**C***), comes from the belief that scalable transformer-based models are highly capable, yet substantially more capable when given tools such as prompt optimization and/or test-time compute, facilitating the model's ability to focus on the most pertinent information for improved predictions. In order to enhance the selection of optimal context for each article, this project incorporates an **Energy-Based Model (EBM)** coupled with **Monte Carlo Sampling**. The EBM is designed as a separate feedforward neural network that assigns a scalar energy value to each article-context pair. Specifically, for a given article $A$ and a generated context $C$, the EBM computes the energy $E(A, C)$ as follows:

$$
E(A, C) = \text{EBM}(f(A), f(C))
$$

where $f(A)$ and $f(C)$ are the embedding vectors of the article and context, respectively. The EBM is implemented as a simple feedforward neural network (Multi-Layer Perceptron) that takes the concatenated embeddings of $A$ and $C$ and outputs a single scalar value representing the energy. I think it's important to note that this implementation is a relatively simple one that draws upon the basic principles of the purpose of an EBM, yet does not dive into more sophisticated approaches such as contrastive divergence or other advanced architectural variants commonly seen in modern EBMs, yet this deliberate simplification allows for efficient context optimization while maintaining computational tractability.

### Energy Computation and Scaling

The energy $E(A, C)$ directly correlates with the MSE of the model's prediction for the given article-context pair. Since a lower MSE indicates a better prediction, it aligns with the EBM's objective of assigning lower energy to more useful contexts. The MSE values are scaled using Min-Max Normalization to ensure numerical stability and effective probability distribution during sampling:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\text{Scaled%20Energy}%20=%20\frac{\text{MSE}%20-%20\text{MSE}_{\text{min}}}{\text{MSE}_{\text{max}}%20-%20\text{MSE}_{\text{min}}%20+%20\epsilon}">
</p>

where $\epsilon$ is a small constant to prevent division by zero. This scaling transforms the energy values to a standardized range, facilitating a more balanced and meaningful probability distribution for sampling.

### Monte Carlo Sampling Based on Energy Values

**Monte Carlo Sampling** is employed to select the most useful contexts for training. The sampling probability $P(C)$ for each context tuple $C$ is defined using the Boltzmann distribution:

$$
P(C) = \frac{e^{-\text{Scaled Energy}(A, C)/T}}{\sum_{i} e^{-\text{Scaled Energy}(A, C_i)/T}}
$$

where $T$ is the temperature parameter that controls the randomness of the sampling process. A lower $T$ biases the selection towards contexts with the lowest energy (highest usefulness), while a higher $T$ introduces more diversity in context selection.

**Sampling Process:**

1. **Energy Evaluation:**  
  For each article $A$ in a batch, the EBM evaluates and assigns energy scores to $k$ generated contexts, where $k$ is defined as the max between the total number of epochs minus the current epoch and 5. This pyramid-down approach ensures that the EBM focuses on capturing the most relevant knowledge during training while maintaining computational efficiency.

2. **Probability Computation:**  
   The scaled energy values are transformed into sampling probabilities using the Boltzmann distribution.
   
3. **Context Selection:**  
   Perform Monte Carlo Sampling using the computed probabilities to select a singular article to pass to the Mixture of Experts for training. Monte Carlo Sampling is performed here as opposed to something like a **greedy selection** (select context with lowest MSE) to promote diversity in context selection, thus mitigating overfitting to a specific subset of contexts.

4. **Integration with Training Loop:**  
  The selected contexts are concatenated with the input data and fed into the model during training iterations. This dynamic selection allows the model to minimize the allocation of resources towards processing the most pertinent information.

### EBM Loss

The **EBM Loss** is designed to train the Energy-Based Model to accurately predict the scaled MSE of each context tuple. By minimizing the discrepancy between the EBM's predicted energies and the actual scaled MSEs, the EBM learns to assign lower energy scores to contexts that result in better prediction performance, allowing for an alignment that ensures Monte Carlo Sampling effectively selects high-quality contexts.

## Online Learning/Catastrophic Forgetting[^11][^12]

Online Learning, a methodology in which sequentially available data is used to update the predictor (the weights of a model) for future data at each step, has been an area of focus underrepresented in asset pricing, much less in the context of small cap stocks. Ergonomically, it caters well to the case wherein the data itself is generated as a function of time, such as a stock price. Yet, it doesn't come without its challenges, notably *catastrophic forgetting*. Catastrophic forgetting is a phenomenon in which neural networks, when paired with a form of continuous learning (such as online learning), become prone to forget how to perform older tasks as new tasks are learned. Biologically, humans generally do not suffer from the same predicament. The way we update our biological neural net doesn't necessarily override the neurons that are responsible for holding together old memories that may be useful in performing a task. Instead, the brain integrates new information by strengthening existing neural pathways and forming new connections, ensuring that old memories remain intact and accessible. This process, known as 'synaptic plasticity', allows for the retention and incorporation of both old and new information efficiently. Many algorithms attempt to replicate this special kind of plasticity-stability balance in ensuring consistency in learned prediction.

Regularization of a model's parameters is one of the most commonly used strategies to mitigate catastrophic forgetting. By penalizing large updates to critical parameters during training, regularization helps ensure that the model retains its previously acquired knowledge while adapting to new information. Methods such as **L2 regularization**, **Synaptic Intelligence**, and **Elastic Weight Consolidation**, among others, constrain parameter updates in an effort to prevent drastic shifts in learned weights, albeit in different ways. Likewise, though more focused on the parameters of the gating network rather than the model's weights, **Expert Routing Regularization** is another form of regularization that adds a further layer of control by encouraging a more balanced selection of experts. This ensures that no expert becomes overly dominant or underutilized during training - both signs of overfitting and generalization loss. More creative approaches, such as the **Memory Replay Buffer**, introduce a complementary mechanism. Instead of relying solely on constraining parameter updates, replay buffers store key historical data samples. During training, the model can revisit these past data points alongside new information, akin to how humans recall and integrate past experiences while learning something new. While the literature on addressing catastrophic forgetting is certainly not scarce, there's no clear consensus on the optimal approach. Thus, the following five approaches have been implemented to test their efficacy:

### Continual Learning with Regularization[^16][^17][^18]

L2 regularization, while differing from other regularization approaches here due to its lack of task-specificity, provides a more general approach to stabilizing essential weights during updates. Continuing from the idea of regularization, Continual Learning with Regularization via L2 regularization aims to stabilize the model's parameters by penalizing large weight updates, thereby encouraging smoother transitions during updates. Unlike other task-specific methods, a penalty is applied across all parameters, regardless of their relevance to past tasks. This makes it a more general form of regularization, helping the model resist drastic weight changes during updates without specifically focusing on critical parameters.

The L2 regularization term adds a quadratic penalty to the loss function, expressed as:

![Equation](https://latex.codecogs.com/png.latex?\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{task}}+\lambda\sum_{i}\theta_i^2)

Where:
- $\mathcal{L}_{\text{task}}$ is the original task-specific loss,
- $\theta_i$ are the model's parameters,
- $\lambda$ is the regularization strength, controlling how much to penalize large weights.

This regularization discourages large parameter values, ensuring the model maintains smoother gradients when adjusting to new data. Though not task-specific, its significance lies in its simplicity and ability to be a baseline, guiding reference to other regularization methods.

### Expert Routing Regularization[^19][^20]

Much like L2 regularization, Expert Routing Regularization is fairly standard procedure for industry-scale models. Attempting to improve the efficiency of expert selection by promoting a balanced load across experts, various approaches to regularizing the expert routing mechanism include:

1. **Load Balancing Loss**: This method aims to evenly distribute data points across all experts by adding a penalty term that encourages a more uniform selection of experts.

2. **Expert Dropout**: Similar to standard dropout, expert dropout randomly drops certain experts during training, which prevents over-reliance on any specific expert.

3. **Entropy Regularization** (focus of this project): This method encourages diversity in expert selection by maximizing the entropy of the routing decisions. The idea is to increase the uncertainty in selecting experts, which leads to a more balanced and dynamic allocation of experts for different data points. The entropy loss is formulated as:

    ![Equation](https://latex.codecogs.com/png.latex?\mathcal{L}_{\text{entropy}}=-\sum_{i=1}^{K}p_i\log{p_i})

where \$p_i$ is the probability of selecting expert $i$.

### Synaptic Intelligence[^18][^21]

Tracking the importance of each parameter during training, Synaptic Intelligence (SI) penalizes updates to critical parameters based on how much they contributed to previous tasks. The importance of each parameter, $\Omega_i$, is calculated using accumulated gradient information:

$\Omega_i = \sum_{t=1}^{T} \frac{\partial \ell_t}{\partial \theta_i} \Delta \theta_i$

The total loss, including SI regularization, is:

![Equation](https://latex.codecogs.com/png.latex?\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{task}}+\lambda\sum_{i}\Omega_i(\theta_i-\theta_i^{\text{old}})^2)

Adapting to new data while preserving important past knowledge. 

Its namesake is derived by how the brain manages learning. Synapses, the connections between neurons, strengthen or weaken over time based on the importance of memories or skills, a process known as *synaptic plasticity*. Similarly, SI helps the model prioritize and protect 'important' parameters from being overwritten, just as the brain retains key memories while still allowing us to learn new information.

### Elastic Weight Consolidation[^17][^18]

Elastic Weight Consolidation (EWC) estimates the importance of each parameter by computing the Fisher Information Matrix, which measures the sensitivity of the model's predictions to changes in each parameter. In the context of sequential tasks, such as Task A (old) and Task B (new), EWC identifies weights important to Task A and penalizes their updates during training on Task B. This approach aims to stay within the low error region for Task A while learning Task B.

**How it Works**:

1. **Compute Fisher Information**:
    - Calculate the Fisher Information $F_i$ for each parameter $\theta_i$:

       $F_i = \mathbb{E}\left[ \left( \frac{\partial \log p(y | x, \theta)}{\partial \theta_i} \right)^2 \right]$
    
    - This quantifies how much the probability of the correct prediction changes with small variations in $\theta_i$.

2. **Regularization Term**:
    - Incorporate the Fisher Information into the loss function to penalize significant changes to important parameters:
  
      ![Equation](https://latex.codecogs.com/png.latex?\mathcal{L}_{\text{total}}%20=%20\mathcal{L}_B(\theta)%20+%20\sum_{i}%20\frac{\lambda}{2}%20F_i%20(\theta_i%20-%20\theta_{A,i}^*)^2)
      
    - Where:
      - ![L_B(θ)](https://latex.codecogs.com/png.latex?\mathcal{L}_B(\theta)): the loss for Task B
      - ![\lambda](https://latex.codecogs.com/png.latex?\lambda): controls the regularization strength
      - ![\theta_{A,i}^*](https://latex.codecogs.com/png.latex?\theta_{A,i}^*): the parameter values after Task A
      
3. **Parameter Update**:
    - During training on Task B, parameters with higher $F_i$ receive larger penalties for changes, thereby protecting essential knowledge from Task A.

4. **Integration with Training Loop**:
    - The EWC regularization term is added to the task-specific loss, guiding the optimizer to make balanced updates that are far more aware of the importance of each parameter.

This method borrows from a Bayesian learning perspective, where EWC leverages the Fisher Information Matrix to regularize the learning of new tasks while retaining important information from previous tasks. By prioritizing the preservation of critical parameters, EWC ensures that the model maintains its performance on older data while effectively adapting to new information.

### Memory Replay Buffer[^18][^22][^23]

Differing from its 'regularization' counterparts, Memory Replay Buffers tackle catastrophic forgetting by revisiting historical data samples during training. The buffer stores a selection of past examples, and when new data is introduced, a mixture of old and new samples are replayed during the training process. This ensures that the model maintains performance on previous tasks while adapting to new information, much like how humans recall past experiences when learning something new.

**How it Works**:

1. **New Data**: At each time step $t$, new data $(x_t, y_t)$ is received and added to the buffer along with its prediction error.

2. **Error-Based Sampling**: The model retrieves a batch of old data $(x_{\text{old}}, y_{\text{old}})$ from the buffer based on their prediction errors and sector performance. Sectors with higher average prediction errors have a higher probability of their samples being replayed.
   
3. **Training**: The model trains on both the new data $(x_t, y_t)$ and the sampled batch from the memory buffer $(x_{\text{old}}, y_{\text{old}})$, allowing it to learn from new information while reinforcing its knowledge of past data.

4. **Updating the Buffer**: The buffer has a fixed size. When new data is added, older data is removed using a Prioritized Sampling strategy. The strategy involves deprioritzing samples with lower prediction errors to make room for more informative examples, maintaining a balance of both recent and older, high-error examples in the buffer.

5. **Buffer Strategies**:
   - **Prioritized Sampling** (focus): Selects important past data based on a specific metric (e.g., importance to model performance; in this case, prediction error).
   - **FIFO**: Perhaps the most basic implementation, wherein the oldest data is replaced when the buffer is full.
   - **Reservoir Sampling**: Maintains a random selection of data.

7. **Synthetic Data Replay** (Optional; not the focus of this project): Another similar approach to mention involves generating synthetic examples instead of replaying real data. This is often beneficial in scenarios where storing all historical data is impractical, yet retaining performance on older tasks is essential.

**Error-Based Sampling**

An Error-Based Sampling strategy is implemented to enhance the effectiveness of the Memory Replay Buffer. In this approach, each sector (e.g., Finance, Technology, Utilities) is treated as a distinct task. The model continuously monitors its prediction performance across these sectors by tracking the average prediction error for each. During the sampling process, sectors with higher average errors are assigned a higher probability of their samples being replayed. This ensures that the model allocates more training resources to sectors that are underperforming, thereby improving its predictive accuracy in those areas. By focusing on sectors with greater prediction challenges, the model becomes more robust and adaptable to handle diverse, new market conditions.

**Fixed Limit on New Data per Batch**

In order to prevent the model from overfitting to new data, a fixed limit is imposed on the amount of new data that can be included in each training batch. Specifically, the number of new data samples is capped (e.g., 8 samples per batch). When a larger batch of new data is available, it is divided into multiple smaller batches that adhere to this limit. For instance, let's say the number of new samples is 44 and the fixed new data limit is 8:

**No Fixed Limit**

- 2 batches with 16 new samples each
- 1 batch with 12 new samples and 4 replayed samples from the Memory Replay Buffer

**Fixed Limit**

- 5 batches with 8 new samples each paired with 8 replayed samples from the Memory Replay Buffer
- 1 batch with 4 new samples paired with 12 replayed samples

The fixed limit approach ensures a consistent balance between new data and replayed data in each batch, helping the model avoid overfitting to recent data and ensuring that past knowledge is continually reinforced. 

### Catastrophic Forgetting Testing

Assessing catastrophic forgetting involves systematically evaluating the model's ability to retain knowledge from previous tasks after incorporating new information. The testing suite is designed to simulate real-world scenarios in financial modeling, where new market data is continually available. In each test, the model is incrementally trained on new tasks and evaluated on its performance on previously learned tasks. The core evaluation metric for each task is the Mean Squared Error (MSE), but additional metrics such as R-squared are also considered to provide a broader view of performance.

#### Testing Setup

1. **Task Definition**: A task represents a specific subset of the data, in this case split into $k$ random sectors (from 'Sector' column). Each task consists of training data and a corresponding test set - constituting an 85/15 split.

2. **Parameter Definitions**: In our case, let's say $k=3$ and our sectors are "Technology", "Finance", and "Utilities". The data from each sector is treated as a distinct task.

3. **Training and Evaluation Process**: For each Catastrophic Forgetting mitigation method (or combination of a them):

    - The model is first trained on the Task 1 dataset (e.g., Technology sector).
    - After completing the training on Task 1, the model’s performance is evaluated on the Task 1 test set to establish a baseline.
    - The model is then updated with the Task 2 dataset (e.g., Finance sector), and its performance is evaluated again on both the Task 1 and Task 2 test sets to observe any degradation (or forgetting) in its performance on Task 1.
    - This process is repeated for Task 3 (e.g., Utilities sector), testing the model’s performance on all previously encountered tasks.

This method incorporates a recursive approach to evaluating the model's ability to maintain performance across previously learned tasks while being updated with new ones. At each training step, the model is updated with new data from a specific sector, and its performance is measured not only on the current task but also on all previously encountered tasks. This ensures that any degradation in performance (i.e., catastrophic forgetting) is tracked in real time across all sectors.

At step $t$, after training on task $T_t$, the model is evaluated on all tasks from $T_1$ to $T_t$, and the total loss function can be expressed as:

![Equation](https://latex.codecogs.com/png.latex?\mathcal{L}_{\text{total}}^{(t)}=\sum_{i=1}^{t}\mathcal{L}(\hat{y}_i^{(t)},y_i))

Where:
- $\mathcal{L}(\hat{y}_i^{(t)}, y_i)$ is the loss on task $i$ after training on task $T_t$,
- $\hat{y}_i^{(t)}$ is the model’s prediction for task $i$ after being updated on task $T_t$,
- $y_i$ is the true value for task $i$,
- $t$ represents the current task being trained on.

The loss for a task is recalculated after training on each subsequent task to detect any increases in error, which would indicate forgetting. This looped evaluation provides insight into how well the model retains knowledge from previous tasks after learning new ones.

By summing losses across all tasks, how the total error changes can be tracked, which helps quantify the extent of catastrophic forgetting. The goal is to minimize the increase in the total loss across all previous tasks as new tasks are introduced.

# Citations

[^1]: Chen, L., Pelger, M., & Zhu, J. (2019, March 11). Deep Learning in Asset Pricing. arXiv.org. https://arxiv.org/abs/1904.00745
[^2]: Fan, J., Ke, Z. T., Liao, Y., & Neuhierl, A. (2022). Structural Deep Learning in Conditional Asset Pricing. SSRN Electronic Journal. https://doi.org/10.2139/ssrn.4117882
[^3]: nbettencourt/SC454k · Datasets at Hugging Face. (n.d.). https://huggingface.co/datasets/nbettencourt/SC454k
[^4]: Dong, Z., Fan, X., & Peng, Z. (2024, February 9). FNSPID: A Comprehensive Financial News Dataset in Time Series. arXiv.org. https://arxiv.org/abs/2402.06698
[^5]: Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017, January 23). Outrageously Large Neural Networks: the Sparsely-Gated Mixture-of-Experts Layer. arXiv.org. https://arxiv.org/abs/1701.06538
[^6]: Samsung. (2020, November 11). [SAIF 2020] Day 1: Energy-Based Models for Self-Supervised Learning - Yann LeCun | Samsung [Video]. YouTube. https://www.youtube.com/watch?v=BqgnnrojVBI
[^7]: Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., & Cobbe, K. (2023, May 31). Let’s Verify Step by Step. arXiv.org. https://arxiv.org/abs/2305.20050
[^8]: AI Explained. (2023, November 24). Q* - Clues to the Puzzle? [Video]. YouTube. https://www.youtube.com/watch?v=ARf0WyFau0A
[^9]: Dao, T. (2023, July 17). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. arXiv.org. https://arxiv.org/abs/2307.08691
[^10]: Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022, May 27). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. arXiv.org. https://arxiv.org/abs/2205.14135
[^11]: Hoi, S. C. H., Sahoo, D., Lu, J., & Zhao, P. (2018, February 8). Online Learning: A Comprehensive Survey. arXiv.org. https://arxiv.org/abs/1802.02871
[^12]: Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., & Wermter, S. (2019). Continual Lifelong Learning with Neural Networks: A Review. Neural Networks, 113, 54–71. https://doi.org/10.1016/j.neunet.2019.01.012
[^13]: Griewank, A., & Walther, A. (2000). Algorithm 799: revolve. ACM Transactions on Mathematical Software, 26(1), 19–45. https://doi.org/10.1145/347837.347846
[^14]: You, Y., Gitman, I., & Ginsburg, B. (2017, August 13). Large Batch Training of Convolutional Networks. arXiv.org. https://arxiv.org/abs/1708.03888
[^15]: Chen, T., Xu, B., Zhang, C., & Guestrin, C. (2016, April 21). Training Deep Nets with Sublinear Memory Cost. arXiv.org. https://arxiv.org/abs/1604.06174
[^16]: Zhao, X., Wang, H., Huang, W., & Lin, W. (2024, June 10). A Statistical Theory of Regularization-Based Continual Learning. arXiv.org. https://arxiv.org/abs/2406.06213
[^17]: Kirkpatrick, James, et al. "Overcoming Catastrophic Forgetting in Neural Networks." 2017. arXiv, https://arxiv.org/pdf/1612.00796
[^18]: Hand, Paul. "Continual Learning and Catastrophic Forgetting." 2020. YouTube, https://www.youtube.com/watch?v=vjaq03IYgSk
[^19]: Lewis, M., Bhosale, S., Dettmers, T., Goyal, N., & Zettlemoyer, L. (2021, March 30). BASE Layers: Simplifying Training of Large, Sparse Models. arXiv.org. https://arxiv.org/abs/2103.16716
[^20]: Fedus, W., Zoph, B., & Shazeer, N. (2021, January 11). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. arXiv.org. https://arxiv.org/abs/2101.03961
[^21]: Zenke, F., Poole, B., & Ganguli, S. (2017, March 13). Continual Learning Through Synaptic Intelligence. arXiv.org. https://arxiv.org/abs/1703.04200
[^22]: Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T. P., & Wayne, G. (2018, November 28). Experience Replay for Continual Learning. arXiv.org. https://arxiv.org/abs/1811.11682
[^23]: Shin, Hanul, et al. "Continual Learning with Deep Generative Replay." 2017. arXiv, https://arxiv.org/pdf/1705.08690
