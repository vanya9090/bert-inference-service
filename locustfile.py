from locust import HttpUser, task, between
import random

TEXTS = [
    "Домашнее задание: Inference Service для rubert-mini-frida",
    "Необходимо реализовать inference service для модели rubert-mini-frida.",
    "На выходе -- работающий код, замеры выбранных метрик для сервиса и отчет.",
    "Часть 1 — Выбор фреймворка",
    "Примеры:  fastapi, flask, aiohttp,"
]

class EmbeddingUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def embed_text(self):
        payload = {"text": random.choice(TEXTS)}
        self.client.post("/embed", json=payload)
