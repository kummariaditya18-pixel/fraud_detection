"""Synthetic data training entry point. Replace load_data() to use real data."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

FEATURES = ["amount", "hour", "category", "distance_km", "txn_last_24h", "is_foreign", "new_device", "card_present", "balance_ratio"]
CATEGORIES = ["Grocery", "Fuel", "Restaurant", "Shopping", "Travel", "Electronics", "Entertainment", "ATM", "Online services"]

def generate(n=60000, seed=42):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"amount": np.clip(rng.lognormal(4.25, .9, n), 5, 50000), "hour": rng.integers(0,24,n),
        "category": rng.integers(0,9,n), "distance_km": np.clip(rng.exponential(12,n),0,500),
        "txn_last_24h": np.clip(rng.poisson(3,n),0,80), "is_foreign": rng.binomial(1,.05,n),
        "new_device": rng.binomial(1,.08,n), "card_present": rng.binomial(1,.65,n),
        "balance_ratio": rng.beta(1.8,5,n)})
    logit = -6.0 + .00008*d.amount + 1.0*(d.hour<5) + .012*d.distance_km + .12*d.txn_last_24h + 1.4*d.is_foreign + 1.35*d.new_device + 1.0*(1-d.card_present) + 2.0*d.balance_ratio + .8*d.category.isin([4,5,8])
    d["is_fraud"] = rng.binomial(1, 1/(1+np.exp(-logit)))
    return d

def load_data(n=60000):
    """Single replacement point for a future real-dataset loader."""
    return generate(n)

def main():
    data=load_data(); X=data[FEATURES]; y=data.is_fraud
    xtr,xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    model=RandomForestClassifier(n_estimators=250,max_depth=14,class_weight="balanced_subsample",n_jobs=-1,random_state=42)
    model.fit(xtr,ytr); pred=model.predict(xte); prob=model.predict_proba(xte)[:,1]
    tn,fp,fn,tp=confusion_matrix(yte,pred).ravel()
    metrics={"accuracy":round(accuracy_score(yte,pred),4),"precision":round(precision_score(yte,pred,zero_division=0),4),"recall":round(recall_score(yte,pred,zero_division=0),4),"f1":round(f1_score(yte,pred,zero_division=0),4),"roc_auc":round(roc_auc_score(yte,prob),4),"confusion_matrix":{"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)},"feature_importances":dict(zip(FEATURES,map(lambda x:round(float(x),4),model.feature_importances_)))}
    Path("model").mkdir(exist_ok=True); joblib.dump({"model":model,"features":FEATURES,"categories":CATEGORIES},"model/fraud_model.joblib")
    Path("model/metrics.json").write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
if __name__ == "__main__": main()
