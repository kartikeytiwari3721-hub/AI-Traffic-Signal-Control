FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    sumo \
    sumo-tools \
    sumo-doc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV SUMO_HOME=/usr/share/sumo
EXPOSE 5000

# Run with gunicorn for Render Web Service compatibility
# Since OpenCV and SUMO need concurrent threads/workers, we run gunicorn with threads
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--threads", "4", "--timeout", "120"]
