FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
COPY pyproject.toml /app/
COPY . /app
RUN pip install --no-cache-dir -r /app/requirements.txt && pip install -e .

EXPOSE 8501

CMD ["streamlit", "run", "insightforge_app.py", "--server.address=0.0.0.0", "--server.port=8501"]

