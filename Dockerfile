FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock
COPY . .
RUN useradd --create-home maintenance && mkdir -p /app/models /app/reports && chown -R maintenance:maintenance /app
USER maintenance
EXPOSE 8000 8501
CMD ["uvicorn", "src.maintenance.api:app", "--host", "0.0.0.0", "--port", "8000"]
