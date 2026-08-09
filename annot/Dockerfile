FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Spaces expect 7860. store.py sits beside app.py, as in review_service.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
