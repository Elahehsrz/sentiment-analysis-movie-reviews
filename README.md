# Sentiment Analysis with Small Language Models and Prompt Engineering

## Overview
This project explores the performance of small language models (SLMs) in sentiment analysis of movie reviews. We compare two quantized models from the Qwen2.5 family, each with different parameter counts: 
- **Qwen2.5-1.5B-Instruct-GGUF (1.5 billion parameters)**
- **Qwen2.5-0.5B-Instruct-GGUF (500 million parameters)**

The goal is to evaluate how different model sizes affect performance, inference speed, and overall efficiency while implementing prompt engineering techniques to enhance accuracy.

## Objectives
- **Prepare a representative data subset** suited for experiments in the given timeframe.
- **Develop and refine prompts** to maximize sentiment analysis accuracy.
- **Implement local inference** using Python for both models.
- **Compare model performance and efficiency** based on various parameters.
- **Analyze results** and document key findings.
- **Visualize results** to facilitate a comprehensive understanding of model behaviors.

## Requirements
Ensure that the following Python packages are installed:
- `datasets`
- `numpy`
- `pandas`
- `matplotlib`
- `llama-cpp-python` (for efficient CPU inference with quantized GGUF models)
- `torch` 
- `transformers`
- `evaluate`

Install the required packages by running:
```bash
pip install -r requirements.txt
```

## Dataset
This project uses a subset of the [IMDB Movie Reviews](https://huggingface.co/datasets/ajaykarthick/imdb-movie-reviews) dataset, which contains 40,000 entries.

## Project Workflow
1. **Data Preparation and Sampling**: Explanation of how the dataset is loaded and a balanced subset is selected.
2. **Prompt Engineering**: Various prompt structures are explored to improve the models' performance.
3. **Model Inference**: Local inference is performed, detailing optimal inference parameters (e.g., temperature, top_p, top_k).
4. **Evaluation Metrics**: Custom metrics are implemented to compare performance.
5. **Visualizations and Analysis**: Graphical representations are created to showcase comparative results.
6. **Error Analysis**: Detailed insights into where the models succeed or fail and potential areas for improvement.

### Repository Structure:

The repository includes a Jupyter notebook `sentiment_analysis.ipynb` and a `design.md` file that explains our design decisions related to data sampling, prompt engineering, and evaluation metrics. 
The repository also contains the `results` folder.

## Running the Notebook
1. Clone the repository or download the notebook.
2. Ensure you have installed all dependencies listed in `requirements.txt`, ideally in a dedicated virtual environment.
3. Run the notebook using Jupyter or any compatible environment to reproduce the results.

