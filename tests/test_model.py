from train_model import FEATURES, generate
from app import score
def test_generator_and_model_sanity():
    data=generate(500,77); assert list(data[FEATURES])==FEATURES; assert .02 < data.is_fraud.mean() < .08
    genuine={"amount":100,"hour":13,"category":0,"distance_km":1,"txn_last_24h":1,"is_foreign":0,"new_device":0,"card_present":1,"balance_ratio":.05}
    fraud={"amount":18000,"hour":2,"category":8,"distance_km":1000,"txn_last_24h":20,"is_foreign":1,"new_device":1,"card_present":0,"balance_ratio":.95}
    out=score([genuine,fraud],save=False); assert not out[0]['is_fraud'] and out[1]['is_fraud']
