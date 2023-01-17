FROM  python:3.8-slim-buster


COPY requirements.txt ./requirements.txt
COPY predictions.py ./predictions.py
ADD model  /model


RUN pip install --upgrade pip \
        && pip install -r requirements.txt

RUN python3 predictions.py