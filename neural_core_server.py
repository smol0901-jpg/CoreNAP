"""
NEURAL_ARCHITECT_PREMIUM++ — Neural Core v4.0
Единый интеллект с разделением компаний, чат-интерфейсом и интеграцией внешних LLM.
"""

import os, json, hashlib, time, random
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI, HTTPException, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import httpx

# ======================== КОНФИГУРАЦИЯ ========================
STATE_DIM = 15
ACTION_DIM = 9
GROWTH_FACTOR = 25 ** (1/10)   # 25x за 10 обучений
REPORT_INTERVAL = 10
MODEL_PATH = "neural_core.pth"

FREE_API_KEY = "NAPP-FREE-2024"
ARCHITECT_URL = "https://neural-architect.netlify.app"

# ======================== НЕЙРОСЕТЬ ===========================
class DynamicDuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2), nn.LayerNorm(hidden_size // 2), nn.ReLU()
        )
        self.value = nn.Linear(hidden_size // 2, 1)
        self.advantage = nn.Linear(hidden_size // 2, action_dim)

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + a - a.mean(dim=1, keepdim=True)

    def expand(self, factor=GROWTH_FACTOR):
        new_hidden = max(int(self.hidden_size * factor), self.hidden_size + 4)
        new_model = DynamicDuelingDQN(STATE_DIM, ACTION_DIM, new_hidden)
        with torch.no_grad():
            # копирование весов с инициализацией
            for old_layer, new_layer in [
                (self.feature[0], new_model.feature[0]),
                (self.feature[3], new_model.feature[3]),
                (self.feature[6], new_model.feature[6])
            ]:
                old_w, old_b = old_layer.weight, old_layer.bias
                new_w, new_b = new_layer.weight, new_layer.bias
                new_w[:old_w.size(0), :old_w.size(1)] = old_w
                if new_w.size(0) > old_w.size(0):
                    new_w[old_w.size(0):] = torch.randn_like(new_w[old_w.size(0):]) * 0.1
                new_b[:old_b.size(0)] = old_b
            new_model.value.weight[:, :self.feature[6].out_features] = self.value.weight
            new_model.advantage.weight[:, :self.feature[6].out_features] = self.advantage.weight
        self.__dict__.update(new_model.__dict__)
        self.hidden_size = new_hidden

# ======================== БУФЕР ОПЫТА =========================
class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, target):
        self.buffer.append((state, target))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, targets = zip(*[self.buffer[i] for i in idxs])
        return np.array(states), np.array(targets)

# ======================== ХРАНИЛИЩЕ КОМПАНИЙ ===================
class CompanyDataStore:
    """Изолированное хранилище данных по компаниям."""
    def __init__(self):
        # company_id -> { "products": {product_id: {...}}, "context": {...}, "chat_history": [...] }
        self.companies = {}

    def get(self, company_id: str):
        if company_id not in self.companies:
            self.companies[company_id] = {
                "products": {},
                "context": {"name": company_id, "metrics": {}, "last_updated": None},
                "chat_history": []
            }
        return self.companies[company_id]

    def update_context(self, company_id, metrics: dict):
        comp = self.get(company_id)
        comp["context"]["metrics"].update(metrics)
        comp["context"]["last_updated"] = datetime.now().isoformat()

# ======================== ОБЩИЙ МОДЕЛЬНЫЙ МЕНЕДЖЕР =============
class GlobalModelManager:
    """Управляет общей нейросетью и обучает её на данных всех компаний."""
    def __init__(self):
        self.model = DynamicDuelingDQN(STATE_DIM, ACTION_DIM)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        self.memory = ReplayBuffer(capacity=50000)  # общий буфер
        self.train_steps = 0

    def learn(self, states, targets):
        for s, t in zip(states, targets):
            self.memory.push(s, t)
        batch = self.memory.sample(128)
        if batch:
            X, Y = batch
            X_t = torch.FloatTensor(X)
            Y_t = torch.FloatTensor(Y)
            pred = self.model(X_t)
            loss = nn.MSELoss()(pred, Y_t)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.train_steps += 1
            if self.train_steps % REPORT_INTERVAL == 0:
                self.model.expand()
                self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
            return loss.item()
        return None

    def predict(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.model(state_t).squeeze().tolist()
        action = int(np.argmax(q_vals))
        return action, q_vals

# ======================== МОДЕЛИ ЗАПРОСОВ ======================
class LearnRequest(BaseModel):
    company_id: str
    product_id: str = "default"
    states: List[List[float]]
    targets: List[List[float]]

class PredictRequest(BaseModel):
    company_id: str
    product_id: str = "default"
    state: List[float]
    use_external: Optional[List[str]] = None

class ChatRequest(BaseModel):
    company_id: str
    message: str
    language: str = "auto"  # "ru", "en", "auto"
    external_ai: str = "deepseek"  # имя зарегистрированной внешней LLM

class ExternalRegisterRequest(BaseModel):
    name: str
    base_url: str
    api_key: str
    model: Optional[str] = None

class SimulateRequest(BaseModel):
    company_id: str
    product_id: str
    initial_state: List[float]
    steps: int = 10

class AnalyzeRequest(BaseModel):
    company_id: str
    product_id: str
    data: List[List[float]]

# ======================== FASTAPI ПРИЛОЖЕНИЕ ====================
app = FastAPI(title="NEURAL_ARCHITECT_PREMIUM++ Neural Core v4", version="4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

global_model = GlobalModelManager()
company_store = CompanyDataStore()
external_ais = {}  # name -> {"client": httpx.AsyncClient, "model": str}

# ======================== ЗАВИСИМОСТИ ==========================
async def verify_license(request: Request, x_api_key: Optional[str] = Header(None)):
    user_agent = request.headers.get("User-Agent", "")
    if "NEURAL_ARCHITECT" not in user_agent and "ASV_PROD" not in user_agent:
        raise HTTPException(403, detail={
            "error": "Только для экосистемы NEURAL_ARCHITECT_PREMIUM++.",
            "url": ARCHITECT_URL
        })
    if x_api_key and x_api_key != FREE_API_KEY and not x_api_key.startswith("NAPP-COM-"):
        raise HTTPException(403, detail="Неверный API-ключ.")
    return True

# ======================== РОУТЫ ================================

@app.get("/")
async def root():
    return {"service": "Neural Core v4", "docs": "/docs"}

@app.post("/learn")
async def learn(req: LearnRequest, authorized: bool = Depends(verify_license)):
    loss = global_model.learn(req.states, req.targets)
    # Обновляем контекст компании (последние метрики)
    if req.states:
        avg_state = np.mean(req.states, axis=0).tolist()
        company_store.update_context(req.company_id, {"avg_state": avg_state, "product": req.product_id})
    return {"status": "ok", "train_steps": global_model.train_steps, "loss": loss}

@app.post("/predict")
async def predict(req: PredictRequest, authorized: bool = Depends(verify_license)):
    action, q_vals = global_model.predict(req.state)
    result = {"action": action, "q_values": q_vals}
    if req.use_external:
        advices = {}
        for name in req.use_external:
            if name in external_ais:
                info = external_ais[name]
                prompt = f"Состояние: {req.state}, рекомендовано действие {action} с Q={q_vals}. Дай совет для компании {req.company_id}."
                try:
                    resp = await info["client"].post("/v1/chat/completions", json={
                        "model": info["model"], "messages": [{"role": "user", "content": prompt}]
                    })
                    advices[name] = resp.json()["choices"][0]["message"]["content"]
                except Exception as e:
                    advices[name] = str(e)
        result["external_advices"] = advices
    return result

@app.post("/chat")
async def chat(req: ChatRequest, authorized: bool = Depends(verify_license)):
    # Собираем контекст компании
    comp = company_store.get(req.company_id)
    context_str = json.dumps(comp["context"], ensure_ascii=False)
    history = "\n".join([f"User: {h['user']}\nAI: {h['ai']}" for h in comp["chat_history"][-5:]])
    prompt = (
        f"Ты — аналитический помощник компании {req.company_id}. "
        f"Отвечай кратко, на {'русском' if req.language=='ru' else 'английском'} языке. "
        f"Контекст компании: {context_str}\n"
        f"История диалога: {history}\n"
        f"Пользователь: {req.message}\nAI:"
    )
    # Используем внешнюю LLM
    if req.external_ai not in external_ais:
        raise HTTPException(400, f"Внешний ИИ '{req.external_ai}' не зарегистрирован.")
    info = external_ais[req.external_ai]
    try:
        resp = await info["client"].post("/v1/chat/completions", json={
            "model": info["model"], "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5
        })
        answer = resp.json()["choices"][0]["message"]["content"]
        # Сохраняем в историю
        comp["chat_history"].append({"user": req.message, "ai": answer, "timestamp": datetime.now().isoformat()})
        return {"response": answer}
    except Exception as e:
        raise HTTPException(500, f"Ошибка LLM: {str(e)}")

@app.post("/external/register")
async def register_external(req: ExternalRegisterRequest):
    client = httpx.AsyncClient(base_url=req.base_url, headers={"Authorization": f"Bearer {req.api_key}"}, timeout=30.0)
    external_ais[req.name] = {"client": client, "model": req.model or "default"}
    return {"status": "registered", "name": req.name}

@app.post("/simulate")
async def simulate(req: SimulateRequest, authorized: bool = Depends(verify_license)):
    history = []
    state = req.initial_state.copy()
    for step in range(req.steps):
        action, q_vals = global_model.predict(state)
        # Простая модель среды: случайное изменение + влияние действия
        state = [s + 0.1 * (action - 4) / 4 + random.uniform(-0.05, 0.05) for s in state]
        history.append({"step": step, "state": state, "action": action})
    return {"history": history}

@app.post("/analyze")
async def analyze(req: AnalyzeRequest, authorized: bool = Depends(verify_license)):
    if not req.data:
        return {"error": "No data"}
    arr = np.array(req.data)
    avg = arr.mean(axis=0).tolist()
    trend = (arr[-1] - arr[0]).tolist() if len(arr) > 1 else [0]*len(arr[0])
    action, q_vals = global_model.predict(avg)
    return {
        "company_id": req.company_id,
        "product_id": req.product_id,
        "average_state": avg,
        "trend": trend,
        "recommended_action": action,
        "q_values": q_vals
    }

# ======================== ЗАПУСК ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🧠 Neural Core v4 (NEURAL_ARCHITECT_PREMIUM++) запущен на порту {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
