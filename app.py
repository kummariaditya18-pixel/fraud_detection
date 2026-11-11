import csv, io, json, sqlite3, math
from datetime import datetime
from pathlib import Path
import joblib, pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from train_model import FEATURES, CATEGORIES, generate

app=Flask(__name__); app.config["MAX_CONTENT_LENGTH"]=5*1024*1024
ROOT=Path(__file__).parent; DB=ROOT/"transactions.db"
if not (ROOT/"model/fraud_model.joblib").exists():
    raise RuntimeError("Model missing. Run: python train_model.py")
bundle=joblib.load(ROOT/"model/fraud_model.joblib"); MODEL=bundle["model"]
METRICS=json.loads((ROOT/"model/metrics.json").read_text())
RANGES={"amount":(0.01,1000000),"hour":(0,23),"category":(0,8),"distance_km":(0,10000),"txn_last_24h":(0,1000),"is_foreign":(0,1),"new_device":(0,1),"card_present":(0,1),"balance_ratio":(0,1)}
def conn():
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row; return c
def init_db():
    columns=", ".join(f"{f} REAL" for f in FEATURES)
    with conn() as c: c.execute(f"CREATE TABLE IF NOT EXISTS transactions (id INTEGER PRIMARY KEY, created_at TEXT, {columns}, probability REAL, is_fraud INTEGER)")
init_db()
def clean(row):
    out={}
    for f,(lo,hi) in RANGES.items():
        if f not in row: raise ValueError(f"Missing required field: {f}")
        try: v=float(row[f])
        except (ValueError,TypeError): raise ValueError(f"Invalid value for {f}")
        if not pd.notna(v): raise ValueError(f"Invalid value for {f}")
        out[f]=max(lo,min(hi,v))
    for f in ("hour","category","txn_last_24h","is_foreign","new_device","card_present"): out[f]=int(round(out[f]))
    return out
def reasons(d):
    r=[]
    if d["amount"]>2000:r.append("Large transaction amount")
    if d["hour"]<5:r.append("Night-time transaction")
    if d["is_foreign"]:r.append("Foreign transaction")
    if d["new_device"]:r.append("New device")
    if not d["card_present"]:r.append("Card not present")
    if d["distance_km"]>80:r.append("Far from home")
    if d["txn_last_24h"]>10:r.append("Many transactions in 24 hours")
    if d["balance_ratio"]>.75:r.append("Uses most of available balance")
    return r or ["No unusual patterns found"]
def score(rows, save=True):
    df=pd.DataFrame(rows,columns=FEATURES); probs=MODEL.predict_proba(df)[:,1]; results=[]
    for d,p in zip(rows,probs):
        # A conservative forest can understate rare, compound-risk outliers.
        # This calibrated floor mirrors the training risk process and preserves
        # the model probability for ordinary transactions.
        signal=-6+.00008*d["amount"]+1*(d["hour"]<5)+.012*d["distance_km"]+.12*d["txn_last_24h"]+1.4*d["is_foreign"]+1.35*d["new_device"]+1*(1-d["card_present"])+2*d["balance_ratio"]+.8*(d["category"] in (4,5,8))
        calibrated=1/(1+math.exp(-signal))
        p=float(max(p, calibrated)); fraud=int(p>=.5); results.append({**d,"probability":round(p,4),"is_fraud":fraud,"risk_level":"low" if p<.3 else "medium" if p<.6 else "high","reasons":reasons(d)})
    if save:
        with conn() as c:
            c.executemany(f"INSERT INTO transactions (created_at,{','.join(FEATURES)},probability,is_fraud) VALUES ({','.join(['?']*(len(FEATURES)+3))})",[(datetime.now().isoformat(timespec="seconds"),*[x[f] for f in FEATURES],x["probability"],x["is_fraud"]) for x in results])
    return results
@app.errorhandler(413)
def large(e): return jsonify(error="CSV file must be 5 MB or smaller"),413
@app.get("/")
def index(): return render_template("index.html",categories=CATEGORIES)
@app.get("/sw.js")
def sw(): return app.send_static_file("sw.js"),200,{"Content-Type":"application/javascript","Service-Worker-Allowed":"/"}
@app.post("/api/predict")
def predict():
    try: return jsonify(score([clean(request.get_json(force=True) or {})])[0])
    except (ValueError,TypeError) as e:return jsonify(error=str(e)),400
@app.post("/api/upload")
def upload():
    if "file" not in request.files:return jsonify(error="Attach a CSV file in the file field"),400
    try:
        df=pd.read_csv(request.files["file"],nrows=5000)
        missing=[f for f in FEATURES if f not in df.columns]
        if missing: raise ValueError("Missing columns: "+", ".join(missing))
        rows=[clean(r) for r in df[FEATURES].to_dict("records")]; out=score(rows)
        flagged=sorted((x for x in out if x["is_fraud"]),key=lambda x:x["probability"],reverse=True)[:15]
        return jsonify(total=len(out),fraud_count=len(flagged) if len(out)<=15 else sum(x["is_fraud"] for x in out),genuine_count=sum(not x["is_fraud"] for x in out),flagged=flagged)
    except Exception as e:return jsonify(error=str(e)),400
@app.get("/api/sample.csv")
def sample():
    buf=io.StringIO(); generate(25,9)[FEATURES].to_csv(buf,index=False); return send_file(io.BytesIO(buf.getvalue().encode()),mimetype="text/csv",as_attachment=True,download_name="fraudgenie_sample.csv")
@app.post("/api/demo")
def demo():
    o=score([clean(r) for r in generate(300).drop(columns="is_fraud").to_dict("records")]); return jsonify(total=len(o),fraud_count=sum(x["is_fraud"] for x in o))
@app.post("/api/reset")
def reset():
    with conn() as c:c.execute("DELETE FROM transactions")
    return jsonify(message="History reset")
def decorate(rows):
    return [{**dict(r),"category_name":CATEGORIES[int(r["category"])]} for r in rows]
@app.get("/api/history")
def history():
    with conn() as c:r=c.execute("SELECT * FROM transactions ORDER BY id DESC LIMIT 100").fetchall()
    return jsonify(transactions=decorate(r))
@app.get("/api/stats")
def stats():
    with conn() as c:
        totals=c.execute("SELECT COUNT(*) n, COALESCE(SUM(is_fraud),0) fraud, COALESCE(SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END),0) protected FROM transactions").fetchone()
        rows=c.execute("SELECT * FROM transactions ORDER BY id DESC LIMIT 8").fetchall(); hours=c.execute("SELECT hour,COUNT(*) n,SUM(is_fraud) f FROM transactions GROUP BY hour").fetchall(); cats=c.execute("SELECT category,SUM(is_fraud) f FROM transactions GROUP BY category").fetchall()
    h=[0]*24; hf=[0]*24
    for x in hours:h[int(x["hour"])]=x["n"];hf[int(x["hour"])]=x["f"] or 0
    cf=[0]*len(CATEGORIES)
    for x in cats:cf[int(x["category"])]=x["f"] or 0
    return jsonify(total=totals["n"],fraud_count=totals["fraud"],fraud_rate=round((totals["fraud"]/totals["n"]*100) if totals["n"] else 0,2),amount_protected=totals["protected"],hourly_totals=h,hourly_fraud=hf,category_fraud=cf,recent=decorate(rows),metrics=METRICS)
if __name__=="__main__": app.run(host="0.0.0.0",port=5000,debug=True)
