# heavy-equipment-classifier
This MLOps pipeline classifies equipment failures across 10,000+ machines using six continuous sensor streams, preventing ₹8-15 lakh in hourly downtime. The end-to-end system covers data validation (Pandera), model training and tuning (MLflow + Optuna), drift monitoring (Evidently), and prediction explainability (SHAP).

## Setup

Create a project-local Python 3.13 virtual environment and install the notebook dependencies:

```bash
python3.13 -m venv .venv313
source .venv313/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install ipykernel
python -m ipykernel install --user --name heavy-equipment-classifier-py313 --display-name "Python (heavy-equipment-classifier, 3.13)"
```

In VS Code, open `predivtive-maintainance.ipynb`, choose **Select Kernel**, and select **Python (heavy-equipment-classifier, 3.13)**. Alternatively, select the interpreter at `.venv313/bin/python` when prompted. On Apple Silicon macOS, install the native OpenMP runtime required by LightGBM:

```bash
brew install libomp
```

To leave the environment, run:

```bash
deactivate
```

## ML-Flow
```bash
source .venv313/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
