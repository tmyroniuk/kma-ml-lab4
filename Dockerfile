FROM python:3.10.11

WORKDIR /app

COPY config /app/config

COPY src/config.py /app/src/
COPY src/preprocessing.py /app/src/
COPY src/utils.py /app/src/
COPY prediction_script.py /app/

COPY model/xgb-v3.json /app/model/
COPY model/preprocessor.pkl /app/model/

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "prediction_script.py"]
