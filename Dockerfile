FROM python:3.10.6-buster

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /
RUN pip install --upgrade pip && pip install -r /requirements.txt

COPY my_model /my_model
COPY main_api.py /main_api.py

CMD uvicorn main_api:app --host 0.0.0.0 --port $PORT
