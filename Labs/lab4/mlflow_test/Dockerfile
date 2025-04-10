FROM python:3.10

RUN pip install --no-cache-dir mlflow

EXPOSE 5000

CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]