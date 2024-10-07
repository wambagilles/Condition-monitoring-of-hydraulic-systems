FROM python:3.9.17-slim


WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--workers=2", "--threads=4 ", "--worker-class=gthread", "--preload", "--timeout=0",  "--bind=0.0.0.0:9696", "app:app"]


