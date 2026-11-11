"""Smoke tests for FraudGenie's public API."""
import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import app

client = app.test_client()
genuine={"amount":500,"hour":14,"category":0,"distance_km":2,"txn_last_24h":1,"is_foreign":0,"new_device":0,"card_present":1,"balance_ratio":.1}
suspicious={"amount":25000,"hour":2,"category":8,"distance_km":350,"txn_last_24h":19,"is_foreign":1,"new_device":1,"card_present":0,"balance_ratio":.94}
def post(path, **kwargs):
    response=client.post(path,**kwargs); assert response.status_code==200,(response.status_code,response.get_json()); return response.get_json()

post("/api/reset")
assert not post("/api/predict",json=genuine)["is_fraud"]
assert post("/api/predict",json=suspicious)["is_fraud"]
assert client.post("/api/predict",json={"amount":"bad"}).status_code==400
assert client.get("/api/sample.csv").status_code==200
csv=b"amount,hour,category,distance_km,txn_last_24h,is_foreign,new_device,card_present,balance_ratio\n500,14,0,2,1,0,0,1,.1\n25000,2,8,350,19,1,1,0,.94\n"
assert post("/api/upload",data={"file":(io.BytesIO(csv),"valid.csv")},content_type="multipart/form-data")["total"]==2
assert client.post("/api/upload",data={"file":(io.BytesIO(b"amount\n1"),"bad.csv")},content_type="multipart/form-data").status_code==400
assert post("/api/demo")["total"]==300
stats=client.get("/api/stats").get_json(); assert stats["total"]>=304 and len(stats["hourly_totals"])==24 and len(stats["category_fraud"])==9
assert client.get("/api/history").status_code==200 and client.get("/").status_code==200 and client.get("/sw.js").status_code==200
post("/api/reset")
print("All endpoint checks passed; genuine and suspicious classifications verified.")
