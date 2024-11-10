### Introduction

In this project, we focus on evaluating the performance of small language models (SLMs) for sentiment analysis tasks using a subset of movie reviews. This document outlines the rationale and methodology behind key decisions related to data sampling, prompt engineering, and evaluation metrics. 

### Data Sampling and Processing
This project uses a subset of the [IMDB Movie Reviews](https://huggingface.co/datasets/ajaykarthick/imdb-movie-reviews) dataset, which contains 40,000 entries. We reviewed the dataset structure and found it contains both training and test sets. Given that the task focuses on prompt engineering rather than model fine-tuning, we have chosen to work almost exclusively with the test set for our analysis, with the exception of a few train-set samples for few-shot learning. This approach allows us to evaluate the models' sentiment analysis capabilities directly and efficiently, in line with the project's goal of prompt optimization.


Using the `get_unique_values()` function, we identified the unique values in the label column, confirming a binary classification structure with labels `0` and `1`. In this setup, `0` denotes positive sentiment, while `1` represents negative sentiment. 


Using the `create_dataframe()` function, we converted the dataset into a DataFrame format to facilitate easier interaction and data manipulation throughout our analysis.



Before proceeding with model evaluation, it was important to verify the balance of the dataset to ensure unbiased performance. We used a custom function called `get_value_counts()` to assess the distribution of classes in the data. This function checks the counts of each value in the `label` column and returns the distribution as a dictionary. We found that the data has a balanced distribution, with **5,000 instances each for classes `0` and `1`**. Given this balance, we decided to proceed with a subset of the data that maintains this proportional distribution to ensure that both classes are equally represented in our analysis.

To understand the distribution of word counts, we plotted histograms and identified outliers. We use the `get_word_count_stats()` function for this. To refine the dataset, we calculated the 10th and 90th percentiles using the `calculate_percentiles()` function. Reviews outside these percentiles were removed with the `remove_rows_by_word_count()` function, ensuring a cleaner and more representative subset. When testing a few instances, we encountered the challenge of token limitations, as many Hugging Face models can only process up to **512 tokens**. To avoid truncation and potential information loss, we filtered reviews to include only those with a maximum of 450 or 420 tokens (depending on the experiment) to leave enough tokens for the classification prompt. The `filter_by_token_count()` function was employed to select reviews that met this criterion.


Since the token-filtered dataset still contained several thousand samples, we used the `balanced_subset()` function to extract an equal number of positive and negative samples, and initially targeted a subset of **1,000 entries**. However, we soon realized that processing this amount of data on a local CPU was impractical for the given time frame.

To address this, we decided to use much smaller **sample size of 4, 20 or 64**. For parameter experimentation, we used a 20-sample subset, and afterwards conducted experiments on prompt efficiency and comparative analysis between the two models with a 64-sample subset. For initial evaluation of the output format when modifying prompts, we used 4 entries of the 64-sample subset.



### Prompt Engineering Decisions


Our approach to prompt engineering for this task was iterative, aiming for continuous refinement to achieve optimal model performance. However, we maintained the same system prompt throughout the analysis.


#### Zero-shot prompting

We began with a straightforward prompt design:

- **Initial Prompts**:
  - **User Prompt**: `f"Classify the following text as positive or negative: {text}"`
  - **System Prompt**: `"You are a sentiment analysis expert. Your task is to classify the sentiment of movie reviews."`



This initial setup included the role and task description in the system prompt and the specific action in the user prompt, enabling the model to understand the classification objective in a zero-shot context. However, we observed that the model frequently repeated the entire review in its response, resulting in unnecessary token generation. To resolve this issue, we adjusted the user prompt to:

  ```python
  prompt = f"Classify the following text as positive or negative: {text}. Output only 'positive' or 'negative'."
  ```
  This change instructed the model to limit its response to only the labels 'positive' or 'negative'. Despite these modifications, the model occasionally produced inconsistent outputs such as "NEGATIVE" or "POSITIVE". Although we normalized the outputs to lowercase during evaluation, we also added an explicit instruction to the prompt:

  
  ```python
  prompt = f"Classify the following text as positive or negative: {text}. Output only 'positive' or 'negative'. Do not use capital letter in your answer."
  ```
  This additional directive improved the consistency of the model's responses.
We refer to this prompt as *prompt_zero_shot* for future reference.


#### Few-shot prompting

To evaluate performance beyond zero-shot prompting, we experimented with a few-shot approach using the smaller model. This involved providing the model with a few labeled examples as context. Initially, we included **4 examples** (2 positive and 2 negative)  from the train set. However, this approach made the prompt longer, leading to issues with the maximum token limit.

To address this, we reduced the context to **1 example from each label** (positive and negative). This ensured that the combined prompt remained under **512 tokens**, maintaining compatibility with the model's token limit while still providing a balanced representation of sentiment examples. We added a few-shot setup with examples formatted as follows:

- *"review"* → **positive**
- *"review"* → **negative**



 ```python
  prompt = f"Classify the sentiment of the following text as positive or negative: {row.review}. Output only 'positive' or 'negative'. Do not use capital letter in your answer. \nI am providing you with a few examples:\n {examples}"
  ```

#### Chain-of-thought prompting

Another step to enhance the model's performance involved incorporating a **chain-of-thought** process by prompting the model to generate reasoning for each prediction. Additionally, we asked the model to assign a confidence score using the following prompt:


  ```python
  prompt = f"Classify the sentiment of the following text as positive or negative: {row.review}. Output only 'positive' or 'negative'. Do not use capital letter in your answer. Explain your reasoning and assign a confidence_score to your prediction. Your output should be a dictionary with 'predicted_label', 'reasoning' and 'confidence_score' as keys."
  ```


This prompt caused an issue with apostrophes and double quotes, resulting in an invalid string literal error. Although we instructed the model to escape apostrophes in the reasoning,

  ```python
  prompt = f"Classify the sentiment of the following text as positive or negative: {row.review}. Output only 'positive' or 'negative'. Do not use capital letter in your answer. Explain your reasoning and assign a confidence_score to your prediction. Your output should be a dictionary with the keys 'predicted_label', 'reasoning', and 'confidence_score'. Escape apostrophes in your reasoning."
  ```

the error persisted, and the model didn't follow the instruction. To troubleshoot the issue, we proceeded by changing the format of the output. Therefore, we prompted to use the delimiter `\t`:


  ```python
  prompt = f"Classify the sentiment of the following text as positive or negative: {row.review}. As a predicted label, output only 'positive' or 'negative'. Do not use capital letter in your answer. Explain your reasoning and assign a float-value confidence_score between 0 and 1. Your output should be a tab-separated (\t) string as follows: <predicted_label>\t<reasoning>\t<confidence_score>."
  ```

This also did not solve the problem, and we continued to encounter the same issue with quotes (invalid string literal error). We then modified the prompt to use the delimiter `|`:



  ```python
  prompt = f"Classify the sentiment of the following text as positive or negative: {row.review}. Output only 'positive' or 'negative'. Do not use capital letters in your answer. Explain your reasoning and assign a confidence_score to your prediction. Your output should be as follows: predicted_label|confidence_score|reasoning"
  ```
This prompt led to good reasoning and classification, but the output always included 'predicted_label: ...|confidence_score: ...|reasoning: ...', and we wanted the output to include only the values while maintaining the same performance. We tried the below prompts, none of which yielded the desired output:


  ```python
  prompt = f"Classify the sentiment of the following text as positive or negative: {row.review}. Output only 'positive' or 'negative'. Do not use capital letters in your answer. Explain your reasoning and assign a confidence_score to your prediction. Your output should be as follows: predicted_label|confidence_score|reasoning. Do not repeat words 'predicted_label' and 'confidence_score' and 'reasoning' in your output."

  prompt = f"Classify the sentiment of the following text as positive or negative: {row.review}. For the classification, output only 'positive' or 'negative'. Also explain your reasoning for the classification. You should separate predicted label and reasoning with a pipe symbol `|`, e.g.: 'positive|This is a positive review because...' or 'negative|This is a negative review because...'. "    

  prompt = f"Classify the sentiment of the following text as positive or negative: {row.review}. First, explain your reasoning, then classify based on your reasoning. For the classification, output only 'positive' or 'negative'. Separate reasoning and classification with a pipe symbol `|`, e.g.: 'This is a positive review because...|positive' or 'This is a negative review because...|negative'. "    
  ```



Finally, we used the initial prompt with the delimiter `|` to get the output and applied the function `replace_key_at_start_or_after_pipe()` to extract the expected values for `predicted_label`, `confidence_score`, and `reasoning` using regex.


  ```python
prompt = f"Classify the sentiment of the following text as positive or negative: {row.review}. Output only 'positive' or 'negative'. Do not use capital letters in your answer. Explain your reasoning and assign a confidence_score to your prediction. Your output should be as follows: predicted_label|confidence_score|reasoning"
  ```

To run the function with this prompt (which contains 59 tokens), we had to remove two rows of the 64-sample subset to meet the maximum token length limit. Therefore, for a fair comparison with the large model, we also evaluate the performance of the latter with 62 samples.


### Parameter Selection

**Temperature**: This parameter controls the randomness of the model’s output. Lower values make the output more deterministic, while higher values increase diversity in the responses. Since lower temperatures reduce randomness and ensure more focused and predictable results, we selected a temperature of **0** for this task.


**Top-p (Nucleus Sampling)**: This parameter dynamically selects tokens from the smallest possible set whose cumulative probability meets or exceeds the threshold `p`. This approach avoids sampling from low-confidence tokens, reducing randomness and irrelevant choices. High values of `top-p` (e.g., 0.9–0.95) retain flexibility for coherent text generation.
   
**Top-k Sampling**: This parameter chooses from the `k` most probable tokens, which is effective but less adaptable across contexts. With a low `k`, sampling is focused on highly probable responses, reducing noise but sometimes missing out on tokens if the probability distribution is flat (i.e., several equally likely options). Smaller `top-k` values (e.g., `k = 5–10`) ensure that only the most probable outputs are considered, reducing the risk of including low-probability options.


For our initial run with 20 samples, we set the two parameters to **0**. However, to observe their impact on the model's performance, we made a change. Since sentiment analysis involves only two possible classes ("positive" or "negative"), we can set top-p to **0.9** or **0.95**. For top-k, we use **5** or **10**. In our setting, the configuration should provide a balance between confidence in output selection and the flexibility required for accurate classification, particularly for binary tasks like sentiment analysis.


To efficiently explore the impact of different inference parameters, we performed a small grid search using two values for each parameter (`top_p_list = [0.8, 0.9]`, `top_k_list = [5, 10]`) and tested them using the smaller model, which was faster to evaluate.


Our experiments showed that adjusting `top_p` and `top_k` did not lead to any differences in the results for the 20-sample subset. Therefore, we decided to continue the evaluation using consistent values of **`0.9` for `top_p`** and **`5` for `top_k`** for both the small and large models for evaluating 64 samples. 



### Evaluation Metrics


To evaluate the performance of our model in the binary text classification task, we selected several key metrics and visualization methods. Since the dataset is balanced between positive and negative classes, **accuracy** was used to measure the overall correctness of the model's predictions. Additionally, we included **precision**, **recall**, and **F1-score**. To visualize performance, we used **bar charts** for all four metrics and **confusion matrices** to highlight true versus false predictions, providing deeper insight into the model's strengths and weaknesses. The **classification report** was also employed to offer a comprehensive summary of the precision, recall, and F1 scores for each class.

In addition to the standard evaluation metrics, we performed an **error analysis** to gain insights into the reasons behind the model's predictions. We examined a few misclassified samples, exploring whether the length of the reviews played a role in these errors. 



## Results and Key Insights

Results with 20 samples and the  *prompt_zero_shot* are reported as follows:




| Model       | Computation Time | Accuracy | F1 Score          | Precision       | Recall |
|-------------|-|---------|-------------------|-----------------|--------|
| Large Model |20.95 min| 0.95     | 0.949874686716792 | 0.9545454545454546 | 0.95   |
| Small Model | 6.26 min|0.9      | 0.898989898989899 | 0.9166666666666667 | 0.9    |

As we can observe, the larger model outperforms the smaller model with 20 samples, while being more than 3x slower. As we can see in the table below, the conducted grid search with the small model using different configurations did not yield any differences for 20 samples: 

| Configuration       | Accuracy | F1 Score          | Precision       | Recall |
|---------------------|----------|-------------------|-----------------|--------|
| top_p_0.8_top_k_5   | 0.9      | 0.898989898989899 | 0.9166666666666667 | 0.9    |
| top_p_0.8_top_k_10  | 0.9      | 0.898989898989899 | 0.9166666666666667 | 0.9    |
| top_p_0.9_top_k_5   | 0.9      | 0.898989898989899 | 0.9166666666666667 | 0.9    |
| top_p_0.9_top_k_10  | 0.9      | 0.898989898989899 | 0.9166666666666667 | 0.9    |



Next, we increased the number of samples to 64. While still a relatively small number, this was manageable for the CPU run, offering better efficiency and time management.


The table below shows the performance of both models using 64 samples, showing that both models achieved the exact same scores:



| Model       | Accuracy | F1 Score          | Precision       | Recall |
|-------------|----------|-------------------|-----------------|--------|
| Large Model |   0.8125   | 0.8108374384236454 | 0.8238866396761133 | 0.8125   |
| Small Model |   0.8125   | 0.8108374384236454 | 0.8238866396761133 | 0.8125   |



We continued the analysis with the smaller model to improve its performance (since it's faster) by applying a few-shot learning approach. The results with the few-shot examples for the smaller model are worse than *prompt_zero_shot*:



|  Accuracy | F1 Score          | Precision       | Recall |
|----------|-------------------|-----------------|--------|
| 0.75    |  0.7333333333333334 | 0.8333333333333333 |0.75    |


The results of the smaller model when employing **chain-of-thought** and **confidence score generation** are as follows:




|  Accuracy | F1 Score          | Precision       | Recall |
|----------|-------------------|-----------------|--------|
| 0.819672131147541  |  0.8146920740127037 | 0.8391608391608392 |0.8135775862068966   |


The results for the large model with 62 samples are as folows:

|  Accuracy | F1 Score          | Precision       | Recall |
|----------|-------------------|-----------------|--------|
| 0.8064516129032258  |   0.8031746031746032 |  0.819078947368421 |0.803125  |



In our error analysis section, we observed a few cases where the model failed and we checked if this is related to the length of the tokens. Given the mean token length of 233.16 and a standard deviation of 97.84, our conclusion is that mispredictions of the model are not linked to a specific token length. 

- The smaller model was faster in terms of computation but had lower performance for the 20-sample subset.
- The larger model was slower (approx. 3 times), but it outperformed the smaller model for the 20-sample subset.
- Adding two few-shot examples did not improve the model's performance
- Adding chain-of-thought and confidence score generation did not improve the performance of the smaller model.




### Challenges and Possible Solutions


One of the main challenges we encountered was the extended computation time when running the experiments on a CPU. The limited processing power of a standard CPU did not allow us to perform on a larger number of samples for this evaluation.   To mitigate this issue, a solution would be to use  a GPU for computation. Running the notebook on platforms like Google Colab or any environment with GPU support can substantially reduce computation time and improve overall efficiency.

Another issue was the maximum token limitations for both models.  To address this challenge, the potential solutions include:

- **Summarization**: Summarizing reviews to ensure they contain fewer than 512 tokens without significant loss of meaning. This approach would require using a more powerful language model capable of producing high-quality summaries.
- **Truncation**: Although truncating movie reviews is an option, it leads to loss of information, potentially impacting the accuracy and comprehensiveness of the analysis. 


In the approach using chain-of-thought processing and confidence-score generation, we observed that the confidence scores were always equal to or higher than **0.9**, and the model generated categorical outputs such as 'predicted labels', 'reasoning', and 'confidence scores' instead of numerical values. This created challenges for evaluation and computing all scores. To address this issue, we decided to drop one row containing these categorical outputs to ensure that only rows with valid, numeric confidence scores were included in the evaluation. 

