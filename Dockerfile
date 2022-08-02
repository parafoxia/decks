FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app
COPY requirements.txt ./
RUN pip install -U pip
RUN pip install -Ur requirements.txt
COPY . .
