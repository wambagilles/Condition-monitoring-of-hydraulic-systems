FROM python:3.9.17-slim


WORKDIR /app

COPY requirements.txt /app/requirements.txt

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "app:app"]


