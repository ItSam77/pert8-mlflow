# Comparing Fine-tuning vs Non-fine-tuning Models with CNN

**Source Dataset:** https://www.kaggle.com/datasets/vencerlanz09/bottle-synthetic-images-dataset

## Quick Overview

This dataset contains many bottle images with 5 different types of bottles:
- Beer Bottles
- Plastic Bottles  
- Soda Bottles
- Water Bottles
- Wine Bottles

The goal is to build a CNN model and compare the performance between fine-tuning and non-fine-tuning approaches. We use MLflow to log the models and track metrics.

## Getting Started

### 1. Clone this repository
```bash
git clone https://github.com/ItSam77/pert8-mlflow.git
cd pert8-mlflow
```

### 2. Setup the virtual environment
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install the dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the Dataset
Download the dataset from the source link and extract it to the `img` folder:
- **Dataset URL:** https://www.kaggle.com/datasets/vencerlanz09/bottle-synthetic-images-dataset
- **Note:** Make sure to download the dataset and put it in the `img` folder

### 5. Prepare the Data
Run the Jupyter notebook to split the data into train, test, and validation sets.
**Note:** Make sure to set your kernel to the virtual environment version.

### 6. Start MLflow UI
Launch the MLflow tracking UI to monitor your experiments:
```bash
mlflow ui
```
The UI will be available at http://localhost:5000

### 7. Train the Models
Run the modeling script to train both models (with and without fine-tuning):
```bash
python modelling.py
```

**Note:** Training might take about an hour with a normal CPU/GPU. Using an external GPU is recommended for faster training.

## Viewing Results

1. After training is complete, navigate to the MLflow UI at http://localhost:5000
2. Click on the **Experiments** tab
3. Select the **Default** experiment
4. You can now compare the results of both models (with and without fine-tuning)

## Project Structure

```
pert8/
├── dataset_split/          # Split dataset (train/val/test)
├── img/                    # Original dataset images
├── mlruns/                 # MLflow experiment logs
├── modelling.py           # Main training script
├── notebook.ipynb         # Data preparation notebook
├── requirements.txt       # Python dependencies
└── README.md             # This file
```