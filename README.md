# MGTF 423 Course Final Project: Text Analysis and Bert Fine-tuning 

![Course](https://img.shields.io/badge/Course-MGTF%20423-blue.svg)
![Status](https://img.shields.io/badge/Status-Completed-green.svg)

Welcome to the code repository for the UCSD - MGTF 423 Data Science for Finance course final project. This project aims to conduct an small - scale but in-depth exploratory analysis of the given text data(10k reports for Fluence Energy, Inc. from Y2022-Y2024) and also try to explore BERT model fine-tuning based on the datasets on a relevant large amount of text strings extracted from 10k files(SEC), loaded from Huggingface

---

## Repository Structure

This repository contains several key components to help you quickly understand and reproduce our work:

-   **`/source`**: Contains the main code for data processing and analysis.
    -   `exploratory_analysis/`: The Jupyter Notebooks for the exploratory text analysis.
    -   `bert_finetuning/`: Contains code for the BERT model fine-tuning, intended for future work.
-   **`/data/raw`**: Stores the original raw PDF data files used in the project.
-   **`/data/derived`**: Stores the pre-processed and cleaned text data (in `.json` format).
-   **`/outputs/graphs`**: Contains charts and visualizations, including frequency bar charts and word cloud maps generated during the EDA(or ETA) process.

---

## Part 1: Exploratory Text Analysis

The core objective of this section is to preprocess and clean the raw text data, and then perform an exploratory analysis on 10k reports downloaded from SEC, for Fluence Energy, Inc. to uncover potential patterns and topics(or themes).

### Key Activities

1.  **Data Preprocessing and Cleaning**: Extracting text from the original PDF files and performing standardization.
2.  **Data Analysis and Visualization**: Generating descriptive statistical charts and maps to analyze text features.
3.  **Interactive Topic Exploration**: Building a standalone web application to interactively explore and analyze the results of the LDA model.

## Part 2: Model Training (BERT Fine-tuning)

This section details the experimental design for BERT model fine-tuning, which is considered a in-progress research for this project. We fine-tuned a pre-trained BERT model on a specific dataset to accomplish a particular task.

### Dataset

* **Source**: We imported the datasets [**financial-reports-sec**] (including three split datasets: train/validation/test, already) by JanosAudran for model fine-tuning.
* **Link**: The complete dataset can be accessed viewing: **`/data/raw/Reports_sec`**.

### Training Parameters

We have defined the following hyperparameter pool for the BERT fine-tuning experiment. The final model will be trained and selected based on these parameters.

```python
config = {
    "model_name": "bert-base-uncased",
    "dataset_path": "[Your path]",
    "sample_ratio": 0.0005,         # Sampling Rate
    "learning_rate": 1e-5,          # Learning Rate
    "train_batch_size": 32,         # Training Batch Size
    "eval_batch_size": 32,          # Evaluation Batch Size
    "num_train_epochs": 5,          # Number of Training Epochs
    "frozen_layers": [              # Freeze the last two layers of the BERT model
        "bert.encoder.layer.10",
        "bert.encoder.layer.11"
    ]
}
```

> **Note**: The relevant model training code is located in the `/source/bert_finetuning/` directory as backup scripts for future work.
