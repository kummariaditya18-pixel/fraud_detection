# FraudGenie

FraudGenie is an installable Flask dashboard that uses a Random Forest model to score individual or CSV transaction records for fraud risk.

## Features

- Instant transaction screening with clear risk reasons
- CSV batch scoring (up to 5 MB / 5,000 rows), demo data, history and analytics
- SQLite-backed audit trail, responsive dark-mode UI, PWA offline shell
- Synthetic training pipeline with metrics and generated application icons

## Tech stack

Python, Flask, SQLite, pandas, scikit-learn, joblib, Chart.js, vanilla HTML/CSS/JS.

## Folder structure

`app.py` is the API/server, `train_model.py` trains the model, `make_icons.py` creates PWA icons, `model/` holds generated artifacts, `templates/` holds the page, and `static/` contains the PWA and UI assets.

## Windows PowerShell setup

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install --only-binary=:all: -r requirements.txt
python make_icons.py
python train_model.py
python app.py
```

Open http://127.0.0.1:5000. To use another device on the same Wi-Fi, find this PC's IPv4 address with `ipconfig`, then open `http://YOUR_IPV4:5000` on the device. Allow Python through Windows Firewall if prompted.

## Deploy to Render

Use build command `pip install -r requirements.txt && python train_model.py` and start command `gunicorn app:app`.

## Real data

Replace the `load_data()` body in `train_model.py`; this is the sole training data entry point. Map Kaggle Credit Card Fraud or PaySim data to the nine documented `FEATURES`, use a binary `is_fraud` target, then rerun the training script.

## Model metrics

After training, current accuracy, precision, recall, F1, ROC-AUC, confusion matrix and feature importances are saved in `model/metrics.json` and displayed in the dashboard.
