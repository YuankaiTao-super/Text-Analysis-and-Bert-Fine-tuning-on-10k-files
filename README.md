# MGTF 495 Course Final Project: Text Analysis and Model Research

![Course](https://img.shields.io/badge/Course-MGTF%20495-blue.svg)
![Status](https://img.shields.io/badge/Status-Completed-green.svg)

Welcome to the code repository for the MGTF 495 course final project. This project aims to conduct an in-depth exploratory analysis of the given text data and explores BERT model fine-tuning as a potential direction for future work.

---

## 📂 Repository Structure

This repository contains several key components to help you quickly understand and reproduce our work:

-   **`/source`**: Contains the main code for data processing and analysis.
    -   `exploratory_analysis.ipynb`: The Jupyter Notebook for the exploratory text analysis.
    -   `bert_finetuning/`: Contains code for the BERT model fine-tuning, intended for future work.
-   **`/data_raw`**: Stores the original raw PDF data files used in the project.
-   **`/data_processed`**: Stores the pre-processed and cleaned text data (in `.json` format).
-   **`/outputs/graphs`**: Contains charts and visualizations generated during the data analysis.
-   **`/outputs/webpage`**: Contains a web-based interface for interactively exploring the topic analysis results.

---

## 📊 Part 1: Exploratory Text Analysis

The core objective of this section is to preprocess and clean the raw text data, and then perform an exploratory analysis using visualization methods to uncover potential patterns and themes.

### Key Activities

1.  **Data Preprocessing and Cleaning**: Extracting text from the original PDF files and performing standardization.
2.  **Data Analysis and Visualization**: Generating descriptive statistical charts to analyze text features.
3.  **Interactive Topic Exploration**: Building a standalone web application to interactively explore and analyze the results of the topic model.

### How to Reproduce

1.  Ensure you have all the necessary libraries installed in your environment (e.g., `pandas`, `matplotlib`, `nltk`).
2.  Run the cells in `/source/exploratory_analysis.ipynb` to reproduce the entire data processing and analysis workflow.
3.  The processed data will be saved in the `/data_processed` directory.
4.  Open the `/outputs/webpage/index.html` file in a browser to interact with the topic analysis results.

---

## 🤖 Part 2: Model Training (BERT Fine-tuning)

This section details the experimental design for BERT model fine-tuning, which is considered a future research direction for this project. The plan is to fine-tune a pre-trained BERT model on a specific dataset to accomplish a particular task.

### Dataset

* **Source**: We plan to use the [**Your Dataset Name Here**] dataset for model fine-tuning.
* **Link**: The complete dataset can be accessed via the following link: `[Insert Data Link Here]`.
* **Processing Script**: The preprocessing script for the dataset is located at `/source/bert_finetuning/prepare_data.py`.

### Training Parameters

We have defined the following hyperparameter pool for the BERT fine-tuning experiment. The final model will be trained and selected based on these parameters.

```python
config = {
    "model_name": "bert-base-uncased",
    "dataset_path": "[Insert path to processed data here]",
    "learning_rate": 2e-5,          # Learning Rate
    "train_batch_size": 16,         # Training Batch Size
    "eval_batch_size": 32,          # Evaluation Batch Size
    "num_train_epochs": 3,          # Number of Training Epochs
    "sampling_rate": 0.8,           # Sampling Rate (if applicable)
    "frozen_layers": [             # Freeze the last two layers of the BERT model
        "bert.encoder.layer.10",
        "bert.encoder.layer.11"
    ]
}
```

> **Note**: The relevant model training code is located in the `/source/bert_finetuning/` directory as an exploration for future work.
