FROM python:3.11-slim
WORKDIR /opt/codeguard
COPY worker.py /opt/codeguard/worker.py
USER 65534:65534
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-I", "/opt/codeguard/worker.py"]
