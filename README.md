# Natural Language for Asset Pricing

Traditional models for asset pricing, such as the Capital Asset Pricing Model (CAPM), the Arbitrage Pricing Theory (APT), and the Black-Scholes model, have relied on linear relationships between risk factors and returns, focusing primarily on financial metrics like market risk and volatility. These models, while foundational, often fail to capture the complex, non-linear interactions present in modern financial markets and broader economic factors.

Recent research, such as [Deep Learning for Asset Pricing by Chen et al. (2023)](https://arxiv.org/abs/1904.00745) and [Structural Deep Learning in Conditional Asset Pricing by Fan et al. (2022)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4117882), demonstrate the effectiveness of deep learning in handling the non-linearities and time-varying dynamics of asset prices. These approaches leverage architectures such as feed-forward networks, LSTMs, and GANs to incorporate a wide range of economic and firm-specific factors, providing a more flexible and data-driven method for predicting future prices and risk premiums.

Building upon an abundance of research in Large Language Models (LLMs) and Mixture-of-Experts (MoE) architectures, this project introduces a transformer-based SparseMoE Language Model tailored for asset price prediction for small cap stocks, wherein the model's final fully connected (dense) layer consists of one output unit (where the output is a singular continuous scalar value), as opposed to a probability distribution. Drawing from extensive textual data related to a stock in the form of articles, news reports, and financial statements, the model predicts the stock price 30 days out.

## Data

The data for this model draws from [SC454k](https://huggingface.co/datasets/nbettencourt/SC454k), a dataset of roughly 454k news articles and press releases related to small cap stocks from nasdaq.com, soon to be paired with market data. The schema of the data looks something like this:

- **Symbol**: the ticker symbol of respective stock (e.g., AAPL)
- **Security**: the full name of stock associated with the ticker symbol (e.g., Apple Inc.)
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

## Model Architecture

**Total Parameters**: Approximately **254 million parameters**, which includes all shared parameters and parameters from all experts across all transformer blocks.

**Active Parameters**: Approximately **84 million parameters** are active during a forward pass. This includes the shared parameters and the parameters from the experts that are selected by the routing mechanism.

## Training Details

- **Loss Function**: Mean Squared Error (MSE) between the predicted and actual stock prices.

  $\text{Loss} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$

- **Optimizer**: AdamW optimizer with a learning rate of $2 \times 10^{-5}$.

- **Batch Size**: 16 sequences per batch.

- **Epochs**: Trained over 20 epochs.

- **# of Experts:** 8

- **# of Active Experts per Token:** 2

- **Tokenizer:** GPT-2

- **Techniques Used**:
  - **Gradient Checkpointing**: Reduces memory usage by recomputing intermediate activations during the backward pass.
  - **Mixed Precision Training**: Utilizes half-precision floating points to speed up training and reduce memory consumption.
  - **FlashAttention 2**: Efficient attention mechanism for handling long sequences.

 As financial analysis is defined by changing markets, it only makes sense to pair it with an architecture that caters well to its inherent modality. The disparity amongst inputs makes this a suitable candidate for a MoE, and provides contribution to an area of research that has previously not been explored in-depth.

 Inspiration for the basic components of this architecture were taken from terrific work of Avinash Sooriyarachchi - taken from his [MakeMoE](https://github.com/AviSoori1x/makeMoE) repo.

 ## Coming Soon

 - **SC454k:** Full implementation of SC454k, complete with market data, comprising of 18 data points across 10 unique days timed around the release of the article. Running across 41 m7g.medium instances currently, stay tuned!
 - **Online Learning**: As new data is always prevalent, would be beneficial to have a model that can update it's parameters 'on the fly' - planning to do some testing on how to address 'catastrophic forgetting', as well.
 - **Layer-wise Learning Rate Decay**: Applying progressively smaller learning rates to earlier layers and higher rates to later layers in attempting to squeeze out the most performance gains in this relatively-small model.
