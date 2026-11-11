"""Train FraudGenie's comparison models on synthetic or supported real data."""
import argparse, json, logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES=["amount","hour","category","distance_km","txn_last_24h","is_foreign","new_device","card_present","balance_ratio"]
CATEGORIES=["Grocery","Fuel","Restaurant","Shopping","Travel","Electronics","Entertainment","ATM","Online services"]
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")

def generate(n=60000, seed=42):
    """Create normal transactions plus three coherent, lightly noisy fraud scenarios."""
    rng=np.random.default_rng(seed); fraud=rng.random(n)<.042; kind=rng.integers(0,3,n)
    # Baseline: normal domestic activity, ordinary distances and ratios.
    d=pd.DataFrame({"amount":np.clip(rng.lognormal(4.25,.72,n),15,12000),"hour":np.clip(rng.normal(14,4.5,n).round(),0,23).astype(int),
      "category":rng.choice([0,1,2,3,6,7],n,p=[.23,.12,.2,.22,.13,.1]),"distance_km":np.clip(rng.exponential(7,n),0,80),
      "txn_last_24h":np.clip(rng.poisson(2.4,n),0,20),"is_foreign":rng.binomial(1,.012,n),"new_device":rng.binomial(1,.035,n),
      "card_present":rng.binomial(1,.78,n),"balance_ratio":rng.beta(1.6,7,n)})
    # Card testing: many small online purchases late at night.
    m=fraud&(kind==0); k=m.sum(); d.loc[m,["amount","hour","category","distance_km","txn_last_24h","is_foreign","new_device","card_present","balance_ratio"]]=np.column_stack((rng.uniform(20,450,k),rng.integers(0,5,k),np.full(k,8),rng.exponential(15,k),rng.integers(10,35,k),rng.binomial(1,.2,k),rng.binomial(1,.45,k),np.zeros(k),rng.uniform(.15,.6,k)))
    # Account takeover: overseas, new device, high-value electronic/travel.
    m=fraud&(kind==1); k=m.sum(); d.loc[m,["amount","hour","category","distance_km","txn_last_24h","is_foreign","new_device","card_present","balance_ratio"]]=np.column_stack((rng.lognormal(8.1,.5,k),rng.integers(0,24,k),rng.choice([4,5,8],k),rng.uniform(300,3000,k),rng.integers(2,12,k),np.ones(k),np.ones(k),rng.binomial(1,.1,k),rng.uniform(.55,.98,k)))
    # Stolen card: remote merchant, atypically far, high balance use.
    m=fraud&(kind==2); k=m.sum(); d.loc[m,["amount","hour","category","distance_km","txn_last_24h","is_foreign","new_device","card_present","balance_ratio"]]=np.column_stack((rng.lognormal(7.3,.55,k),rng.integers(0,24,k),rng.choice([3,4,5,8],k),rng.uniform(90,900,k),rng.integers(4,18,k),rng.binomial(1,.25,k),rng.binomial(1,.3,k),np.zeros(k),rng.uniform(.7,.99,k)))
    d["is_fraud"]=fraud.astype(int)
    # Small label noise keeps the task realistic without destroying the pattern.
    flip=rng.random(n)<.004; d.loc[flip,"is_fraud"]=1-d.loc[flip,"is_fraud"]
    return d

def load_data(path=None):
    if not path:return generate()
    raw=pd.read_csv(path); cols=set(raw.columns)
    if {"Time","Amount","Class"}.issubset(cols):
        # Kaggle credit-card data lacks merchant semantics; derive compatible proxies.
        v=lambda name: raw[name] if name in raw else 0
        out=pd.DataFrame({"amount":raw.Amount.clip(0,1_000_000),"hour":(raw.Time/3600%24).astype(int),"category":np.abs(v("V1").rank(pct=True)*9).astype(int).clip(0,8),"distance_km":np.abs(v("V2"))*20,"txn_last_24h":np.abs(v("V3"))*2,"is_foreign":(v("V4")>v("V4").median()).astype(int),"new_device":(v("V5")>v("V5").median()).astype(int),"card_present":(v("V6")<v("V6").median()).astype(int),"balance_ratio":v("V7").rank(pct=True),"is_fraud":raw.Class.astype(int)})
    elif {"isFraud","amount","step"}.issubset(cols):
        typ=raw.get("type",pd.Series("CASH_IN",index=raw.index)); out=pd.DataFrame({"amount":raw.amount,"hour":raw.step%24,"category":typ.map({"PAYMENT":0,"CASH_OUT":7,"TRANSFER":4,"DEBIT":1,"CASH_IN":2}).fillna(3).astype(int),"distance_km":(typ=="TRANSFER").astype(int)*300,"txn_last_24h":raw.groupby(raw.get("nameOrig",raw.index)).cumcount().clip(0,100),"is_foreign":0,"new_device":0,"card_present":(~typ.isin(["CASH_IN","CASH_OUT"])).astype(int),"balance_ratio":(raw.amount/(raw.get("oldbalanceOrg",1).replace(0,1))).clip(0,1),"is_fraud":raw.isFraud.astype(int)})
    else: raise ValueError("Unsupported data schema. Use Kaggle Credit Card Fraud or PaySim CSV.")
    return out[FEATURES+["is_fraud"]].replace([np.inf,-np.inf],np.nan).fillna(0)

def metrics(y,p,threshold):
    pred=(p>=threshold).astype(int); return {"accuracy":round(float(accuracy_score(y,pred)),4),"precision":round(float(precision_score(y,pred,zero_division=0)),4),"recall":round(float(recall_score(y,pred,zero_division=0)),4),"f1":round(float(f1_score(y,pred,zero_division=0)),4),"roc_auc":round(float(roc_auc_score(y,p)),4),"pr_auc":round(float(average_precision_score(y,p)),4)}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--data"); args=parser.parse_args(); data=load_data(args.data)
    x=data[FEATURES]; y=data.is_fraud; xtr,xte,ytr,yte=train_test_split(x,y,test_size=.2,stratify=y,random_state=42)
    models={"Logistic Regression":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=1000,class_weight="balanced",C=1))]),"Random Forest":RandomForestClassifier(n_estimators=250,max_depth=16,min_samples_leaf=2,class_weight="balanced_subsample",n_jobs=-1,random_state=42),"HistGradientBoosting":HistGradientBoostingClassifier(max_iter=250,max_leaf_nodes=31,l2_regularization=1.0,random_state=42)}
    cv=StratifiedKFold(5,shuffle=True,random_state=42); comparison={}; fitted={}
    for name,model in models.items():
        logging.info("Cross-validating %s",name); prob=cross_val_predict(model,xtr,ytr,cv=cv,method="predict_proba",n_jobs=-1)[:,1]
        comparison[name]=metrics(ytr,prob,.5); model.fit(xtr,ytr); fitted[name]=model
    best=max(comparison,key=lambda k:comparison[k]["pr_auc"]); model=fitted[best]; validation=model.predict_proba(xte)[:,1]
    candidates=np.linspace(.05,.95,181); threshold=float(max(candidates,key=lambda t:f1_score(yte,validation>=t)))
    final=metrics(yte,validation,threshold); tn,fp,fn,tp=confusion_matrix(yte,validation>=threshold).ravel(); fpr,tpr,_=roc_curve(yte,validation); precision,recall,_=precision_recall_curve(yte,validation)
    base=xtr.median(numeric_only=True); perm=permutation_importance(model,xte,yte,n_repeats=5,scoring="average_precision",random_state=42,n_jobs=-1)
    payload={"comparison":comparison,"best_model":best,"chosen_threshold":round(threshold,4),"dataset_size":int(len(data)),"fraud_rate":round(float(y.mean()),4),**final,"confusion_matrix":{"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)},"roc_curve":{"fpr":fpr.round(5).tolist(),"tpr":tpr.round(5).tolist()},"pr_curve":{"precision":precision.round(5).tolist(),"recall":recall.round(5).tolist()},"feature_importances":dict(zip(FEATURES,perm.importances_mean.round(5).tolist()))}
    Path("model").mkdir(exist_ok=True); joblib.dump({"model":model,"features":FEATURES,"categories":CATEGORIES,"baseline":base.to_dict(),"threshold":threshold},"model/fraud_model.joblib"); Path("model/metrics.json").write_text(json.dumps(payload,indent=2))
    print("\nModel comparison (5-fold cross-validation)"); print(pd.DataFrame(comparison).T[["accuracy","precision","recall","f1","roc_auc","pr_auc"]].to_string()); print(f"\nBest: {best}; threshold: {threshold:.2f}"); print(json.dumps(final,indent=2))
if __name__=="__main__":main()
