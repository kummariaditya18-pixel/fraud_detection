import io
from app import app, THRESHOLD

def client(): return app.test_client()
GENUINE={"amount":120,"hour":14,"category":0,"distance_km":2,"txn_last_24h":1,"is_foreign":0,"new_device":0,"card_present":1,"balance_ratio":.08}
FRAUD={"amount":15000,"hour":2,"category":8,"distance_km":900,"txn_last_24h":15,"is_foreign":1,"new_device":1,"card_present":0,"balance_ratio":.9}
def test_prediction_explanation_and_threshold():
    c=client(); c.post('/api/reset'); a=c.post('/api/predict',json=GENUINE); b=c.post('/api/predict',json=FRAUD)
    assert a.status_code==b.status_code==200; assert not a.json['is_fraud']; assert b.json['is_fraud']; assert b.json['probability']>=THRESHOLD; assert len(b.json['factors'])==5
def test_bad_input_upload_and_monitoring():
    c=client(); assert c.post('/api/predict',json={'amount':'bad'}).status_code==400
    assert c.post('/api/upload',data={'file':(io.BytesIO(b'amount\n1'),'bad.csv')},content_type='multipart/form-data').status_code==400
    assert c.post('/api/simulate').status_code==200; assert c.get('/api/alerts').status_code==200; assert c.get('/api/stats?threshold=.3').status_code==200
def test_history_and_static_assets():
    c=client(); assert c.get('/').status_code==200; assert c.get('/sw.js').status_code==200; assert c.get('/api/history/export').status_code==200; assert c.get('/api/sample.csv').status_code==200
