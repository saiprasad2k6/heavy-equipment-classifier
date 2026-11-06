# heavy-equipment-classifier
This MLOps pipeline classifies equipment failures across 10,000+ machines using six continuous sensor streams, preventing ₹8-15 lakh in hourly downtime. The end-to-end system covers data validation (Pandera), model training and tuning (MLflow + Optuna), drift monitoring (Evidently), and prediction explainability (SHAP).

## Notebook description

`predivtive-maintainance.ipynb` implements the complete predictive-maintenance workflow. It loads the training, current, and stress-test datasets; validates their schemas with Pandera; explores failure and sensor distributions; encodes categorical features; and balances the training data with SMOTE. It then trains and compares Logistic Regression, Random Forest, XGBoost, and LightGBM models, tracks experiments with MLflow, tunes the best XGBoost model with Optuna, and saves the final model and label encoder.

The notebook also evaluates distribution drift in the current and stress batches using Evidently and generates HTML reports. Finally, it uses SHAP to identify the most influential sensor features for each failure class. Generated model artifacts are stored in `model/`, while visualizations and monitoring reports are stored in `reports/`.

## Project layout

```text
data/
  train.csv       Training data
  current.csv     Current production batch
  stress.csv      Stress-test batch
model/            Generated model and label encoder artifacts
reports/          Generated EDA, drift, and SHAP reports
mlflow/           Local MLflow tracking database
predivtive-maintainance.ipynb
requirements.txt
```

## Setup

The notebook is tested with Python 3.13. From the repository root, create and activate a virtual environment:

```bash
python3.13 -m venv .venv313
source .venv313/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Apple Silicon macOS, install the native OpenMP runtime before running LightGBM:

```bash
brew install libomp
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name heavy-equipment-classifier-py313 --display-name "Python (heavy-equipment-classifier, 3.13)"
```

In VS Code, open `predivtive-maintainance.ipynb`, choose **Select Kernel**, and select **Python (heavy-equipment-classifier, 3.13)**. Alternatively, select `.venv313/bin/python` as the interpreter when prompted.

## Run the notebook

Run all cells from top to bottom. The notebook reads the CSV files from `data/` and creates these directories automatically:

- `model/best_model.pkl`
- `model/label_encoder.pkl`
- `reports/eda_distributions.png`
- `reports/drift_current.html`
- `reports/drift_stress.html`
- `reports/shap_per_class.png`

The notebook also logs model-selection and Optuna runs to the local MLflow database at `mlflow/mlflow.db`.

## View MLflow

With the virtual environment active, start the MLflow UI from the repository root:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5000
```

Open <http://127.0.0.1:5000> in a browser. Stop the server with `Ctrl+C`.

## Deactivate the environment

```bash
deactivate
```

## Troubleshooting

- If VS Code does not show the kernel, reload the window and select the registered `Python (heavy-equipment-classifier, 3.13)` kernel again.
- If LightGBM fails to load on Apple Silicon macOS, verify that `libomp` is installed with Homebrew.
- Run MLflow from the repository root so the relative database path resolves to `mlflow/mlflow.db`.
