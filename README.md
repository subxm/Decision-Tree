# Decision Tree Classifier and Regressor

A comprehensive project implementing Decision Tree models for classification and regression tasks using the IRIS dataset.

## Overview

This project demonstrates the implementation and comparison of Decision Tree models for both classification and regression problems. It includes Jupyter notebooks for detailed analysis and a Streamlit web application for interactive model exploration.

## Project Structure

- `Desicion-tree-classifier.ipynb` - Jupyter notebook implementing Decision Tree Classifier
- `Desicion-tree-regressor.ipynb` - Jupyter notebook implementing Decision Tree Regressor
- `streamlit.py` - Interactive Streamlit application for the Decision Tree Regressor
- `IRIS.csv` - IRIS dataset used for training and evaluation
- `requirements.txt` - Python dependencies

## Features

- Decision Tree Classification model on IRIS dataset
- Decision Tree Regression model for predicting petal length
- Interactive Streamlit application for model exploration
- Model evaluation with performance metrics (MSE, MAE, R2 Score)
- Feature importance analysis
- Hyperparameter tuning examples

## Requirements

- Python 3.7+
- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib

## Installation

1. Clone the repository:

```bash
git clone https://github.com/subxm/Decision-Tree.git
cd Decision-Tree
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Streamlit Application

Start the interactive web application:

```bash
streamlit run streamlit.py
```

The application will open in your browser at `http://localhost:8501`

### Jupyter Notebooks

For detailed model analysis and implementation details:

- Open `Desicion-tree-classifier.ipynb` for classification examples
- Open `Desicion-tree-regressor.ipynb` for regression examples

## Dataset

The project uses the IRIS dataset (IRIS.csv), which contains:

- 150 samples of iris flowers
- 4 features: Sepal Length, Sepal Width, Petal Length, Petal Width
- 3 target classes: Setosa, Versicolor, Virginica

## Model Details

### Decision Tree Classifier

Classifies iris flowers into one of three species based on flower measurements.

### Decision Tree Regressor

Predicts the petal length based on other flower measurements.

Configuration:

- Criterion: squared_error
- Max depth: 6
- Max features: log2
- Splitter: best

## Metrics

The models are evaluated using:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R-squared (R2) Score

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.
