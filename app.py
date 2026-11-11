"""FraudGenie Flask application: scoring, monitoring, cases and analytics APIs."""
import io, json, logging, math, os, sqlite3, time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
import joblib, pandas as pd
from flask import Flask, jsonify, render_template, request, send_file, session
from werkzeug.security import check_password_hash, generate_password_hash
from train_model import FEATURES, CATEGORIES, generate

ROOT=Path(__file__).parent; DB=ROOT/os.getenv("DATABASE_PATH","transactions.db")
app=Flask(__name__); app.config.update(MAX_CONTENT_LENGTH=int(os.getenv("MAX_UPLOAD_MB","5"))*1024*1024,SECRET_KEY=os.getenv("SECRET_KEY","fraudgenie-development-key"),SESSION_COOKIE_HTTPONLY=True,SESSION_COOKIE_SAMESITE="Lax")
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"),format="%(asctime)s %(levelname)s %(message)s")
MODEL_DATA=joblib.load(ROOT/"model/fraud_model.joblib"); MODEL=MODEL_DATA["model"]; THRESHOLD=float(MODEL_DATA["threshold"]); BASELINE=MODEL_DATA["baseline"]
METRICS=json.loads((ROOT/"model/metrics.json").read_text()); AUTH=os.getenv("AUTH_ENABLED","false").lower()=="true"; LIMITS=defaultdict(deque)
RANGES={"amount":(.01,1_000_000),"hour":(0,23),"category":(0,8),"distance_km":(0,10_000),"txn_last_24h":(0,1_000),"is_foreign":(0,1),"new_device":(0,1),"card_present":(0,1),"balance_ratio":(0,1)}
def db(): c=sqlite3.connect(DB);c.row_factory=sqlite3.Row;return c
def migrate():
    cols=", ".join(f"{f} REAL" for f in FEATURES)
    with db() as c:
        c.execute(f"CREATE TABLE IF NOT EXISTS transactions (id INTEGER PRIMARY KEY,created_at TEXT,{cols},probability REAL,is_fraud INTEGER,status TEXT DEFAULT 'New',note TEXT DEFAULT '')")
        existing={r[1] for r in c.execute("PRAGMA table_info(transactions)")}
        if "status" not in existing:c.execute("ALTER TABLE transactions ADD COLUMN status TEXT DEFAULT 'New'")
        if "note" not in existing:c.execute("ALTER TABLE transactions ADD COLUMN note TEXT DEFAULT ''")
migrate()
def auth_required(fn):
    @wraps(fn)
    def wrapped(*a,**k):
        if AUTH and not session.get("authenticated"):return jsonify(error="Authentication required"),401
        return fn(*a,**k)
    return wrapped
def limit(max_calls=30,window=60):
    def deco(fn):
        @wraps(fn)
        def wrapped(*a,**k):
            q=LIMITS[(request.remote_addr,request.endpoint)]; now=time.time()
            while q and q[0]<now-window:q.popleft()
            if len(q)>=max_calls:return jsonify(error="Rate limit exceeded; try again shortly"),429
            q.append(now);return fn(*a,**k)
        return wrapped
    return deco
@app.after_request
def security(response):
    response.headers.update({"X-Content-Type-Options":"nosniff","X-Frame-Options":"DENY","Referrer-Policy":"strict-origin-when-cross-origin","Content-Security-Policy":"default-src 'self'; script-src 'self' https://cdnjs.cloudflare.com 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'"});return response
@app.errorhandler(413)
def too_large(e):return jsonify(error="CSV file must be 5 MB or smaller"),413
def clean(row):
    out={}
    for f,(lo,hi) in RANGES.items():
        if f not in row:raise ValueError(f"Missing required field: {f}")
        try:v=float(row[f])
        except (ValueError,TypeError):raise ValueError(f"Invalid value for {f}")
        if not math.isfinite(v):raise ValueError(f"Invalid value for {f}")
        out[f]=max(lo,min(hi,v))
    for f in ("hour","category","txn_last_24h","is_foreign","new_device","card_present"):out[f]=int(round(out[f]))
    return out
def plain_reasons(d):
    checks=[("amount",2000,"Large transaction amount"),("hour",5,"Night-time transaction"),("is_foreign",.5,"Foreign transaction"),("new_device",.5,"New device"),("distance_km",80,"Far from home"),("txn_last_24h",10,"Many transactions in 24 hours"),("balance_ratio",.75,"Uses most of available balance")]
    r=[label for key,cut,label in checks if (d[key]<cut if key=="hour" else d[key]>cut)];
    if not d["card_present"]:r.append("Card not present")
    return r or ["No unusual patterns found"]
def explanations(frame, probabilities):
    """Model-agnostic local baseline substitution, batched for fast CSV scoring."""
    deltas={}
    for feature in FEATURES:
        altered=frame.copy(); altered[feature]=BASELINE[feature]
        deltas[feature]=probabilities-MODEL.predict_proba(altered)[:,1]
    result=[]
    for i in range(len(frame)):
        factors=[{"feature":f,"label":f.replace("_"," ").title(),"impact_pct":round(abs(float(deltas[f][i]))*100,1),"direction":"increases" if deltas[f][i]>=0 else "reduces"} for f in FEATURES]
        result.append(sorted(factors,key=lambda x:x["impact_pct"],reverse=True)[:5])
    return result
def score(rows,save=True):
    frame=pd.DataFrame(rows,columns=FEATURES); probs=MODEL.predict_proba(frame)[:,1]; factors=explanations(frame,probs); results=[]
    for d,p,why in zip(rows,probs,factors):
        p=float(p); results.append({**d,"probability":round(p,4),"is_fraud":int(p>=THRESHOLD),"risk_level":"low" if p<THRESHOLD*.6 else "medium" if p<THRESHOLD else "high","reasons":plain_reasons(d),"factors":why})
    if save:
        with db() as c:c.executemany(f"INSERT INTO transactions(created_at,{','.join(FEATURES)},probability,is_fraud,status,note) VALUES ({','.join(['?']*(len(FEATURES)+5))})",[(datetime.now().isoformat(timespec="seconds"),*[x[f] for f in FEATURES],x["probability"],x["is_fraud"],"New","") for x in results])
    return results
def decorated(rows):
    return [{**dict(r),"category_name":CATEGORIES[int(r["category"])],"risk_level":"low" if r["probability"]<THRESHOLD*.6 else "medium" if r["probability"]<THRESHOLD else "high"} for r in rows]
@app.get("/")
def index():return render_template("index.html",categories=CATEGORIES,auth_enabled=AUTH)
@app.get("/sw.js")
def sw():return app.send_static_file("sw.js"),200,{"Content-Type":"application/javascript","Service-Worker-Allowed":"/"}
@app.post("/api/login")
def login():
    user=os.getenv("ADMIN_USERNAME","admin"); password=os.getenv("ADMIN_PASSWORD","admin123"); incoming=request.get_json(silent=True) or {}
    valid=(incoming.get("username")==user and (incoming.get("password")==password or check_password_hash(os.getenv("ADMIN_PASSWORD_HASH",generate_password_hash(password)),incoming.get("password",""))))
    if not valid:return jsonify(error="Invalid credentials"),401
    session["authenticated"]=True;return jsonify(message="Signed in")
@app.post("/api/logout")
def logout():session.clear();return jsonify(message="Signed out")
@app.post("/api/predict")
@limit()
@auth_required
def predict():
    try:return jsonify(score([clean(request.get_json(force=True) or {})])[0])
    except (ValueError,TypeError) as e:return jsonify(error=str(e)),400
@app.post("/api/upload")
@limit(8)
@auth_required
def upload():
    if "file" not in request.files:return jsonify(error="Attach a CSV file in the file field"),400
    try:
        frame=pd.read_csv(request.files["file"],nrows=5000); missing=[f for f in FEATURES if f not in frame]
        if missing:raise ValueError("Missing columns: "+", ".join(missing))
        out=score([clean(x) for x in frame[FEATURES].to_dict("records")]); flagged=sorted([x for x in out if x["is_fraud"]],key=lambda x:x["probability"],reverse=True)[:15]
        return jsonify(total=len(out),fraud_count=sum(x["is_fraud"] for x in out),genuine_count=sum(not x["is_fraud"] for x in out),flagged=flagged)
    except Exception as e:return jsonify(error=str(e)),400
@app.get("/api/sample.csv")
def sample():
    s=io.StringIO();generate(25,9)[FEATURES].to_csv(s,index=False);return send_file(io.BytesIO(s.getvalue().encode()),mimetype="text/csv",as_attachment=True,download_name="fraudgenie_sample.csv")
@app.post("/api/demo")
@auth_required
def demo():
    out=score([clean(r) for r in generate(300).drop(columns="is_fraud").to_dict("records")]);return jsonify(total=len(out),fraud_count=sum(x["is_fraud"] for x in out))
@app.post("/api/simulate")
@auth_required
def simulate():
    row=generate(1,int(time.time()*1000)%2**32).drop(columns="is_fraud").iloc[0].to_dict();return jsonify(score([clean(row)])[0])
@app.post("/api/reset")
@auth_required
def reset():
    with db() as c:c.execute("DELETE FROM transactions")
    return jsonify(message="History reset")
@app.get("/api/history")
def history():
    page=max(1,int(request.args.get("page",1))); size=min(100,max(1,int(request.args.get("size",100))));where=[];args=[]
    for key,col in (("status","status"),("category","category"),("fraud","is_fraud")):
        if request.args.get(key) not in (None,""):where.append(f"{col}=?");args.append(request.args[key])
    if request.args.get("search"):where.append("(CAST(id AS TEXT) LIKE ? OR created_at LIKE ?)");args += [f"%{request.args['search']}%"]*2
    clause=(" WHERE "+" AND ".join(where)) if where else ""
    with db() as c: total=c.execute("SELECT COUNT(*) FROM transactions"+clause,args).fetchone()[0]; rows=c.execute("SELECT * FROM transactions"+clause+" ORDER BY id DESC LIMIT ? OFFSET ?",args+[size,(page-1)*size]).fetchall()
    return jsonify(transactions=decorated(rows),total=total,page=page,pages=math.ceil(total/size) if total else 1)
@app.get("/api/history/export")
def export_history():
    with db() as c:rows=decorated(c.execute("SELECT * FROM transactions ORDER BY id DESC").fetchall())
    out=io.StringIO();pd.DataFrame(rows).to_csv(out,index=False);return send_file(io.BytesIO(out.getvalue().encode()),mimetype="text/csv",as_attachment=True,download_name="fraudgenie_history.csv")
@app.get("/api/alerts")
def alerts():
    with db() as c:rows=c.execute("SELECT * FROM transactions WHERE is_fraud=1 ORDER BY id DESC LIMIT 200").fetchall(); counts=dict(c.execute("SELECT status,COUNT(*) FROM transactions WHERE is_fraud=1 GROUP BY status").fetchall())
    return jsonify(alerts=decorated(rows),counts=counts,false_alarm_rate=round(counts.get("False alarm",0)/max(1,sum(counts.values()))*100,2))
@app.patch("/api/alerts/<int:transaction_id>")
@auth_required
def update_alert(transaction_id):
    data=request.get_json(silent=True) or {}; status=data.get("status"); note=str(data.get("note","")[:500])
    if status not in ("New","Under review","Confirmed fraud","False alarm"):return jsonify(error="Invalid alert status"),400
    with db() as c:
        if not c.execute("UPDATE transactions SET status=?,note=? WHERE id=? AND is_fraud=1",(status,note,transaction_id)).rowcount:return jsonify(error="Alert not found"),404
    return jsonify(message="Alert updated")
@app.get("/api/stats")
def stats():
    sensitivity=float(request.args.get("threshold",THRESHOLD)); sensitivity=max(.1,min(.9,sensitivity))
    with db() as c:
        allrows=c.execute("SELECT * FROM transactions").fetchall(); recent=c.execute("SELECT * FROM transactions ORDER BY id DESC LIMIT 8").fetchall()
    vals=decorated(allrows); pred=[x["probability"]>=sensitivity for x in vals]; original=[bool(x["is_fraud"]) for x in vals]; tp=sum(a and b for a,b in zip(pred,original)); fp=sum(a and not b for a,b in zip(pred,original)); fn=sum(not a and b for a,b in zip(pred,original))
    h=[0]*24;hf=[0]*24; cf=[0]*len(CATEGORIES)
    for x,p in zip(vals,pred):h[int(x["hour"])]+=1;hf[int(x["hour"])] += int(p);cf[int(x["category"])] += int(p)
    return jsonify(total=len(vals),fraud_count=sum(pred),fraud_rate=round(sum(pred)/max(1,len(vals))*100,2),amount_protected=sum(x["amount"] for x,p in zip(vals,pred) if p),hourly_totals=h,hourly_fraud=hf,category_fraud=cf,recent=decorated(recent),metrics=METRICS,threshold=THRESHOLD,sensitivity=sensitivity,tradeoff={"precision":round(tp/max(1,tp+fp),3),"recall":round(tp/max(1,tp+fn),3)})
if __name__=="__main__":app.run(host="0.0.0.0",port=int(os.getenv("PORT","5000")),debug=os.getenv("FLASK_DEBUG","false").lower()=="true")
