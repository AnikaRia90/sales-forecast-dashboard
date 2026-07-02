FROM python:3.10-slim

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy all source code
COPY app/ /code/app/
COPY train.py /code/train.py
COPY requirements.txt /code/requirements.txt

# Copy the data folder (all 5 CSVs baked in)
COPY data/ /code/data/

# Copy trained model if it already exists
COPY models/ /code/models/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
