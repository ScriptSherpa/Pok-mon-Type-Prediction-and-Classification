

# Pokémon Type Prediction and Classification

## Overview

This repository contains the implementation of a Pokémon classification project, aimed at predicting whether a Pokémon is **Legendary** or not based on its attributes, as well as predicting its **type** (e.g., Fire, Water, Grass, etc.) using machine learning and neural networks.

This project involves two main tasks:

1. **Legendary vs Non-Legendary Classification** using Random Forest.
2. **Pokémon Type Prediction** using a Feedforward Neural Network (Keras).

The dataset is based on Pokémon attributes, including stats such as attack, defense, speed, and more.

---

## Table of Contents

1. [Project Description](#project-description)
2. [Dataset](#dataset)
3. [Technologies Used](#technologies-used)
4. [Getting Started](#getting-started)
5. [Usage](#usage)
6. [Model Evaluation](#model-evaluation)
7. [Contributing](#contributing)
8. [License](#license)

---

## Project Description

### **Task 1: Legendary Pokémon Classification**

In this task, we build a machine learning model to predict whether a Pokémon is **Legendary** or not based on its attributes (e.g., attack, defense, speed, etc.). We use the **Random Forest Classifier** to train the model and evaluate its performance on a test dataset.

### **Task 2: Pokémon Type Prediction**

For the second task, we predict the **type** of a Pokémon (e.g., Fire, Water, Grass) using a **Feedforward Neural Network** built with Keras. This is a multi-class classification problem where we predict the primary type (`type1`) of the Pokémon based on its stats.

---

## Dataset

The dataset used in this project contains various Pokémon attributes, including:

* `name`: The name of the Pokémon.
* `type1`: The primary type of the Pokémon (target for classification).
* `type2`: The secondary type of the Pokémon (not used in this model).
* `attack`: Attack stat of the Pokémon.
* `defense`: Defense stat of the Pokémon.
* `speed`: Speed stat of the Pokémon.
* `hp`: Health points of the Pokémon.
* `special_attack`: Special attack stat.
* `special_defense`: Special defense stat.
* `height`: Height of the Pokémon.
* `weight`: Weight of the Pokémon.
* `is_legendary`: A binary column indicating if the Pokémon is Legendary.

This dataset is preprocessed to handle missing values, drop irrelevant columns, and encode categorical variables as necessary.

---

## Technologies Used

* **Python**: Programming language used for the entire project.
* **Pandas**: Data manipulation and analysis.
* **Scikit-learn**: For machine learning models (Random Forest).
* **Keras**: Neural network creation and training.
* **Matplotlib/Seaborn**: Data visualization and evaluation (Confusion Matrix heatmap).
* **TensorFlow**: Backend for neural network training.

---

## Getting Started

To get started with this project locally, follow these steps:

### Prerequisites

* Install Python 3.x.
* Install necessary libraries:

```bash
pip install pandas scikit-learn tensorflow matplotlib seaborn
```

### Clone the Repository

```bash
git clone https://github.com/yourusername/pokemon-classification.git
cd pokemon-classification
```

### Dataset

Ensure you have the **Pokémon dataset** (e.g., `pokemon_data.csv`) in the root directory of this project.

---

## Usage

### 1. **Legendary Pokémon Classification**:

Run the following script to train the **Random Forest** model and evaluate its performance.

```bash
python legendary_classification.py
```

This will:

* Train the Random Forest model.
* Output accuracy, confusion matrix, and classification report.
* Display a heatmap of the confusion matrix.

### 2. **Pokémon Type Prediction**:

Run the following script to train the **Neural Network** model using Keras.

```bash
python type_prediction.py
```

This will:

* Train the feedforward neural network on the preprocessed data.
* Output the model accuracy and loss during training.
* Evaluate the model on the test set.

---

## Model Evaluation

### **Legendary Pokémon Classification**

* **Accuracy**: Measures the percentage of correctly predicted labels.
* **Confusion Matrix**: Displays true vs predicted labels.
* **Classification Report**: Precision, recall, and F1-score for both classes (Legendary and Non-Legendary).

### **Pokémon Type Prediction**

* **Accuracy**: Measures the percentage of correctly predicted Pokémon types.
* **Confusion Matrix**: Evaluates classification performance for each Pokémon type.
* **Loss/Accuracy Graphs**: Shows the progression of model training.

---

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request. Any improvements, bug fixes, or suggestions are welcome!

To contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes and commit them.
4. Push your changes and create a pull request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Note:

Feel free to adjust the specific filenames and paths according to your project setup. If you have more models, results, or steps you'd like to add, you can expand the sections.

Good luck with your project! Let me know if you need further assistance.
