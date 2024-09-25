# Natural Language for Asset Pricing

Traditional models for asset pricing, such as the Capital Asset Pricing Model (CAPM), the Arbitrage Pricing Theory (APT), and the Black-Scholes model, have relied on linear relationships between risk factors and returns, focusing primarily on financial metrics like market risk and volatility. These models, while foundational, often fail to capture the complex, non-linear interactions present in modern financial markets and broader economic factors.

Recent research, such as [Deep Learning for Asset Pricing by Chen et al. (2023)](https://arxiv.org/abs/1904.00745) and [Structural Deep Learning in Conditional Asset Pricing by Fan et al. (2022)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4117882), demonstrate the effectiveness of deep learning in handling the non-linearities and time-varying dynamics of asset prices. These approaches leverage architectures such as feed-forward networks, LSTMs, and GANs to incorporate a wide range of economic and firm-specific factors, providing a more flexible and data-driven method for predicting future prices and risk premiums.

Building upon an abundance of research in Large Language Models (LLMs) and Mixture-of-Experts (MoE) architectures, this project introduces a transformer-based SparseMoE Language Model tailored for asset price prediction for small cap stocks, wherein the model's final fully connected (dense) layer consists of one output unit (where the output is a singular continuous scalar value), as opposed to a probability distribution. Drawing from extensive textual data related to a stock in the form of articles, news reports, and financial statements, the model predicts the stock price 30 days out.

## Data

The data for this model draws from [SC454k](https://huggingface.co/datasets/nbettencourt/SC454k), a dataset of roughly 454k news articles and press releases related to small cap stocks from nasdaq.com, soon to be paired with market data. The schema of the data looks something like this:

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
  
**Scraping/Pairing Market Data**

Scraping this dataset was conducted across 10 EC2 t2.large instances across two parts: scraping links, then the content of these links. Puppeteer, a headless Chrome browser in JavaScript, was used for lightweight, quick scraping. Data was subsequently cleaned with a combination of PySpark and NumPy. Pairing this with market data was made available through Wharton Research Data Services and their extensive collection of financial datasets. One collection of these datasets is the Trade and Quote Millisecond dataset (TAQ/MSEC), consisting of millisecond level access to the tick-by-tick trade and quote data of all activity within the U.S. National Market System. The process of pairing this pricing data with the timestamped news article/press release data through their API was magnitudes more tricky than meets the eye. The issue comes down to one thing: latency. The granularity of the API call can only be day-level, meaning a simple call to get the pricing data for say Apr. 3, 2020 will return around 10-15 thousand records of data, and because multiple prices are being retrieved for each news article/press release (I'm grabbing the price of the stock 4 days before, 2 days before, all the way to 30 days after with more granularity in between, totaling 18 different prices across 10 unique days), this is a large amount of data to pair, even with highly-optimized C++ code. Thus, the pairing algorithm went through a number of iterations to optimize efficiency such as promoting a more indexed-based approach to finding dates, a stringent approach to clearing out unused memory, and customized bash scripts for easy deployment of multiple EC2 instances. Ultimately, totaling around 1200 hours of compute across 41 (yes, 41) m7g.medium instances.

Inspiration for this dataset was taken from [FNSPID: A Comprehensive Financial News Dataset in Time Series](https://arxiv.org/abs/2402.06698).

## Key Features

**Sparse Mixture-of-Experts (MoE) Architecture**: An architecture in which $k$ experts are divvied up to solve a task, a popular technique as of recent, boasting similar performance yet less compute compared to their monolithic counterparts. To illustrate why this might be useful, let's think about the case of next token prediction. When handing the prompt “Tell me about President Eisenhower’s first inaugural address” to an LLM, little to no activation is present amongst the majority of the model’s neurons. In this instance, the model will not activate neurons tailored for subjects that are irrelevant towards the prompt. To name a few, neurons focused on technical computing, abstract mathematical theories, and deep scientific concepts, among many others, will stay inactive as their expertise holds, at most, a tangential connection with the prompt. This is, quite frankly, extremely inefficient. When dealing with a task that's already computationally expensive and faced with a shortage of GPUs, this inefficiency only compounds the issues. Thus, MoEs attempt to curtail this inefficiency by creating a number of models that each specialize in a certain task.

**16k Context Window**: A relatively large context window was chosen to ensure the model has access to as much relevant information as possible. For a given input, the current article, along with its metadata (title, publication, author, etc.), are included in the context. Additionally, the previous 10 articles related to the stock are provided. Building on this further, stock prices from 96 hours, 48 hours, and 24 hours prior to the article's release, as well as the price at the time of the article's release, are included. This comprehensive data allows the model to consider both historical articles and market trends when making predictions.

**Custom Training Pipeline:** Optimized for training on NVIDIA A100 GPUs, employing techniques like mixed precision and gradient checkpointing for efficient resource utilization.

**FlashAttention 2**: Implemented for efficient and scalable attention computation, enabling the model to handle long sequences effectively.

**Online Learning**: Designed to continuously adapt to new data streams, ensuring the model remains up-to-date with the latest market trends and information. By leveraging techniques such as Synaptic Intelligence, Memory Replay Buffers, Fisher Information Regularization, and others, the model effectively mitigates catastrophic forgetting, maintaining its predictive accuracy over time while incorporating new insights.

## Model Architecture

**Total Parameters**: Approximately **254 million parameters**, which includes all shared parameters and parameters from all experts across all transformer blocks.

**Active Parameters**: Approximately **84 million parameters** are active during a forward pass. This includes the shared parameters and the parameters from the experts that are selected by the routing mechanism.

## Training Details

- **Loss Function**: Mean Squared Error (MSE) between the predicted and actual stock prices.

  $\text{Loss} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$

- **Optimizer**: AdamW optimizer with a base learning rate of $2 \times 10^{-5}$. Layer-wise learning rate decay is applied with a decay factor of $0.95$ per layer, starting from the deepest layers. The embedding layers and regression head use the base learning rate, while deeper layers have progressively smaller learning rates due to the decay.

- **Batch Size**: 16 sequences per batch.

- **Epochs**: Trained over 20 epochs.

- **# of Experts:** 8

- **# of Active Experts per Token:** 2

- **Tokenizer:** GPT-2

- **Techniques Used**:
  - **Gradient Checkpointing**: Reduces memory usage by recomputing intermediate activations during the backward pass.
  - **Mixed Precision Training**: Utilizes half-precision floating points to speed up training and reduce memory consumption.
  - **FlashAttention 2**: Efficient attention mechanism for handling long sequences.
  - **Layer-wise Learning Rate Decay (LLRD)**: Ensures more stable updates by applying smaller learning rates to lower layers and higher rates to upper layers, improving convergence.

 As financial analysis is defined by changing markets, it only makes sense to pair it with an architecture that caters well to its inherent modality. The disparity amongst inputs makes this a suitable candidate for a MoE, and provides contribution to an area of research that has previously not been explored in-depth.

 Inspiration for the basic components of this architecture were taken from terrific work of Avinash Sooriyarachchi - taken from his [MakeMoE](https://github.com/AviSoori1x/makeMoE) repo.

## Online Learning/Catastrophic Forgetting

Online Learning, a methodology in which sequentially available data is used to update the predictor (the weights of a model) for future data at each step, has been an area of focus underrepresented in asset pricing, much less in the context of small cap stocks. Ergonomically, it caters well to the case wherein the data itself is generated as a function of time, such as a stock price. Yet, it doesn't come without its challenges, notably *catastrophic forgetting*. Catastrophic forgetting is a phenomenon in which neural networks, when paired with a form of continuous learning (such as online learning), become prone to forget how to perform older tasks as new tasks are learned. Biologically, humans generally do not suffer from the same predicament. The way we update our biological neural net doesn't necessarily override the neurons that are responsible for holding together old memories that may be useful in performing a task. Instead, the brain integrates new information by strengthening existing neural pathways and forming new connections, ensuring that old memories remain intact and accessible. This process, known as 'synaptic plasticity', allows for the retention and incorporation of both old and new information efficiently. Many algorithms attempt to replicate this special kind of plasticity-stability balance in ensuring consistency in learned prediction.

Regularization of a model's parameters is one of the most commonly used strategies to mitigate catastrophic forgetting. By penalizing large updates to critical parameters during training, regularization helps ensure that the model retains its previously acquired knowledge while adapting to new information. Methods such as L2 regularization, Synaptic Intelligence, and Fisher Information Regularization, among others, constrain parameter updates in an effort to prevent drastic shifts in learned weights, albeit in different ways. Likewise, though more focused on the parameters of the gating network rather than the model's weights, Expert Routing Regularization is another form of regularization that adds a further layer of control by encouraging a more balanced selection of experts. This ensures that no expert becomes overly dominant or underutilized during training - both signs of overfitting and generalization loss. More creative approaches, such as the Memory Replay Buffer, introduce a complementary mechanism. Instead of relying solely on constraining parameter updates, replay buffers store key historical data samples. During training, the model can revisit these past data points alongside new information, akin to how humans recall and integrate past experiences while learning something new. While the literature on addressing catastrophic forgetting is certainly not scarce, there's no clear consensus on the optimal approach. Thus, the following five approaches have been implemented to test their efficacy:

**Continual Learning with Regularization**: *Coming Soon*. L2 regularization, while differing from other regularization approaches here due to its lack of task-specificity, provides a more general approach to stabilizing essential weights during updates. Continuing from the idea of regularization, Continual Learning with Regularization via L2 regularization aims to stabilize the model's parameters by penalizing large weight updates, thereby encouraging smoother transitions during updates. Unlike other task-specific methods, a penalty is applied across all parameters, regardless of their relevance to past tasks. This makes it a more general form of regularization, helping the model resist drastic weight changes during updates without specifically focusing on critical parameters.

The L2 regularization term adds a quadratic penalty to the loss function, expressed as:

![Equation](https://latex.codecogs.com/png.latex?\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{task}}+\lambda\sum_{i}\theta_i^2)

Where:
- $\mathcal{L}_{\text{task}}$ is the original task-specific loss,
- $\theta_i$ are the model's parameters,
- $\lambda$ is the regularization strength, controlling how much to penalize large weights.

This regularization discourages large parameter values, ensuring the model maintains smoother gradients when adjusting to new data. Though not task-specific, its significance lies in its simplicity and ability to be a baseline, guiding reference to other regularization methods.

**Expert Routing Regularization**: *Coming Soon*. Much like L2 regularization, Expert Routing Regularization is fairly standard procedure for industry-scale models. Attempting to improve the efficiency of expert selection by promoting a balanced load across experts, various approaches to regularizing the expert routing mechanism include:

1. **Load Balancing Loss**: This method aims to evenly distribute data points across all experts by adding a penalty term that encourages a more uniform selection of experts.

2. **Expert Dropout**: Similar to standard dropout, expert dropout randomly drops certain experts during training, which prevents over-reliance on any specific expert.

3. **Entropy Regularization** (focus of this project): This method encourages diversity in expert selection by maximizing the entropy of the routing decisions. The idea is to increase the uncertainty in selecting experts, which leads to a more balanced and dynamic allocation of experts for different data points. The entropy loss is formulated as:

    ![Equation](https://latex.codecogs.com/png.latex?\mathcal{L}_{\text{entropy}}=-\sum_{i=1}^{K}p_i\log{p_i})

where \$p_i$ is the probability of selecting expert $i$.

**Synaptic Intelligence**: Tracking the importance of each parameter during training, Synaptic Intelligence (SI) penalizes updates to critical parameters based on how much they contributed to previous tasks. The importance of each parameter, $\Omega_i$, is calculated using accumulated gradient information:

$\Omega_i = \sum_{t=1}^{T} \frac{\partial \ell_t}{\partial \theta_i} \Delta \theta_i$

The total loss, including SI regularization, is:

![Equation](https://latex.codecogs.com/png.latex?\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{task}}+\lambda\sum_{i}\Omega_i(\theta_i-\theta_i^{\text{old}})^2)

Adapting to new data while preserving important past knowledge. 

Its namesake is derived by how the brain manages learning. Synapses, the connections between neurons, strengthen or weaken over time based on the importance of memories or skills, a process known as *synaptic plasticity*. Similarly, SI helps the model prioritize and protect 'important' parameters from being overwritten, just as the brain retains key memories while still allowing us to learn new information.

**Fisher Information Regularization**: *Coming Soon*. Fisher Information Regularization estimates the importance of each parameter by computing the Fisher Information Matrix, which measures the sensitivity of the model's predictions to changes in each parameter. By penalizing updates to parameters with high Fisher Information, the model preserves critical knowledge from previous tasks.

### **How it Works**:

1. **Compute Fisher Information**:
    - Calculate the Fisher Information $F_i$ for each parameter $\theta_i$:
    
      $F_i = \mathbb{E}\left[ \left( \frac{\partial \log p(y | x, \theta)}{\partial \theta_i} \right)^2 \right]$
    
    - This measures how much the probability of the correct prediction changes with small variations in $\theta_i$.

2. **Regularization Term**:
    - Incorporate the Fisher Information into the loss function to penalize significant changes to important parameters:
    
      ![Equation](https://latex.codecogs.com/png.latex?\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{task}}+\lambda\sum_{i}F_i(\theta_i-\theta_i^{\text{old}})^2)
    
    - Here, $\mathcal{L}_{\text{task}}$ is the original loss, $\lambda$ controls the strength of regularization, and $\theta_i^{\text{old}}$ represents the parameter values from previous tasks.

3. **Parameter Update**:
    - During training, parameters with higher $F_i$ values receive larger penalties for changes, discouraging significant updates and thus protecting essential knowledge.

4. **Integration with Training Loop**:
    - The Fisher Information regularization term is added to the task-specific loss, guiding the optimizer to make balanced updates that respect the importance of each parameter.

The approach prioritizes preserving critical knowledge, ensuring that parameters conducive for previous tasks are less likely to be altered, thus maintaining performance on older data.

**Memory Replay Buffers**: *Coming Soon*. Differing from its 'regularization' counterparts, Memory Replay Buffers tackle catastrophic forgetting by revisiting historical data samples during training. The buffer stores a selection of past examples, and when new data is introduced, a mixture of old and new samples are replayed during the training process. This ensures that the model maintains performance on previous tasks while adapting to new information, much like how humans recall past experiences when learning something new.

### **How it Works**:

1. **New Data**: At each time step \$t$, new data $(x_t, y_t)$ is received and added to the buffer.
   
2. **Old Data**: The model retrieves a random batch of old data $(x_{\text{old}}, y_{\text{old}})$ from the buffer.

3. **Training**: The model trains on both the new data $(x_t, y_t)$ and a sampled batch from the memory buffer $(x_{\text{old}}, y_{\text{old}})$, allowing the model to learn from new information while reinforcing its knowledge of past data.

4. **Updating the Buffer**: The buffer has a fixed size. When new data is added, older data may be removed, typically using a FIFO strategy or a priority-based mechanism. This maintains a balance of both recent and older examples in the buffer.

5. **Buffer Strategies**:
   - **FIFO** (focus): Oldest data is replaced when the buffer is full.
   - **Reservoir Sampling**: Maintains a random selection of data.
   - **Prioritized Sampling**: Selects important past data based on a specific metric (e.g., importance to model performance).

6. **Synthetic Data Replay** (Optional; not the focus of this project): Another similar approach to mention involves generating synthetic examples instead of replaying real data. This is often beneficial in scenarios where storing all historical data is impractical, yet retaining performance on older tasks is essential.

Simply: By replaying old data or generating synthetic samples, the model is less likely to overwrite critical knowledge with new information.

### Catastrophic Forgetting Testing

Assessing catastrophic forgetting involves systematically evaluating the model's ability to retain knowledge from previous tasks after incorporating new information. The testing suite is designed to simulate real-world scenarios in financial modeling, where new stock market data is continually incoming. In each test, the model is incrementally trained on new tasks and evaluated on its performance on previously learned tasks. The core evaluation metric for each task is the Mean Squared Error (MSE), but additional metrics such as R-squared are also considered to provide a broader view of performance.

#### Testing Setup

1. **Task Definition**: A task represents a specific subset of the data, in this case split into $k$ sectors. Each task consists of training data and a corresponding test set - constituting an 85/15 split.

    - In our case, let's say $k=3$ and our sectors are "Technology", "Finance", and "Utilities". The data from each sector is treated as a distinct task.

**Training and Evaluation Process**:

For each Catastrophic Forgetting mitigation method (or combination of a them):

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

By summing losses across all tasks, we can track how the total error changes, which helps quantify the extent of catastrophic forgetting. The goal is to minimize the increase in the total loss across all previous tasks as new tasks are introduced.

# Coming Soon

 - **SC454k:** Full implementation of SC454k, complete with market data, comprising of 18 data points across 10 unique days timed around the release of the article. Running across 41 m7g.medium instances currently, stay tuned!
 - **Online Learning**: As new data is always prevalent, would be beneficial to have a model that can update it's parameters 'on the fly' - planning to do some testing on how to address 'catastrophic forgetting', as well.
