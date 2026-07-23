FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=5550
EXPOSE 5550

# shell form (no brackets) so $PORT actually gets substituted at runtime
CMD gunicorn server:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 180