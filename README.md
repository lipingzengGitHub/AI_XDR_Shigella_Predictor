# XDR Shigella AI Predictor

A PyTorch-based deep learning pipeline to predict antibiotic resistance profiles of XDR-Shigella strains based on genomic features.

## Features

- Train a deep learning model to classify antibiotic resistance from genomic data
- Evaluate model performance (AUC calculation)
- Save trained models and visualize AUC results
- Mock data generator included for testing

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Prepare your data

Use `test_data_generator.py` to generate mock test data, or prepare your own data:

```bash
python test_data_generator.py
```

### Step 2: Run the predictor

```bash
python XDR_Shigella_AI_Predictor.py
```

### Output

- AUC scores: `outputs/auc_scores.txt`
- Trained model: `models/trained_model.pth`
- AUC plot: `outputs/auc_plot.png`

## Project Structure

- `XDR_Shigella_AI_Predictor.py`: Main model pipeline
- `test_data_generator.py`: Generate synthetic dataset
- `data/`: Input files
- `outputs/`: Prediction results
- `models/`: Trained model files


