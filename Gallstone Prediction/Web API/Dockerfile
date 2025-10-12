FROM python:3.8

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

ENTRYPOINT ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "80"]
