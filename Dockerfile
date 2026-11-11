FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt
COPY . .
RUN python make_icons.py && python train_model.py
EXPOSE 5000
CMD ["gunicorn","--bind","0.0.0.0:5000","app:app"]
