# Kuzushiji Character Classification

This project implements a machine learning pipeline to classify Kuzushiji (cursive Japanese) characters from the Kuzushiji-MNIST dataset. It explores the effectiveness of **Histogram of Oriented Gradients (HOG)** features combined with **K-Nearest Neighbors (KNN)**, and uses a **Genetic Algorithm (GA)** to optimize the model's hyperparameters.

## Overview

The project performs the following key steps:
1.  **Data Loading**: Loads the Kuzushiji-MNIST dataset using `deeplake`.
2.  **Exploratory Data Analysis (EDA)**: Visualizes random samples and class distributions.
3.  **Feature Extraction**: Implements HOG to capture local gradients and edge patterns, which are crucial for handwriting recognition.
4.  **Hyperparameter Optimization**: Uses a Genetic Algorithm to find the best combination of:
    -   Kernel type (Sobel, Prewitt, Scharr)
    -   Distance metric (Cosine, Manhattan, Euclidean)
    -   Weights (Uniform, Distance)
    -   Random Projection dimensions
    -   Number of neighbors ($k$)
5.  **Benchmarking**: Compares the performance of the optimized KNN model against Logistic Regression, Decision Trees, and Naive Bayes, using both raw pixel data and HOG features.

## Key Findings

-   **HOG Features**: Significantly improve classification performance compared to raw pixel data by capturing structural morphology.
-   **KNN + HOG**: Achieved the highest accuracy (~90.29%), outperforming other classifiers.
-   **Genetic Algorithm**: Successfully identified optimal hyperparameters (e.g., using Sobel kernel, Consine metric).

## Requirements

-   Python 3.8+
-   See `requirements.txt` for dependencies.

## Installation

1.  Clone the repository.
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This requires `deeplake<4`.*

## Usage

Run the main script to execute the pipeline:

```bash
python main.py
```

The script will:
-   Download/Load the dataset.
-   Display EDA plots.
-   Run the Genetic Algorithm for optimization.
-   Save evolution history to `history.json`.
-   Display analysis plots for the GA process.
-   Print the final classification report and benchmark results.

## License

This project uses the Kuzushiji-MNIST dataset.
