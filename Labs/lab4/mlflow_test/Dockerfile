FROM python:3.10

WORKDIR /Users/skier/MSDS/Spring2/Spring2-MSDS-MLOps/labs/lab4/mlflow_test

RUN pip install --no-cache-dir mlflow

EXPOSE 5000

CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]