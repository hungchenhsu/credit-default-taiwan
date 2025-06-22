# 基底映像：官方 Python + ML 基礎
FROM python:3.11-slim

# 安装 libgomp（OpenMP runtime），让 lightgbm 可以正常加载
RUN apt-get update && \
    apt-get install -y libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# 工作目錄
WORKDIR /app

# 複製需求與專案進容器
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src
COPY ./api ./api
COPY ./artifacts ./artifacts
COPY ./mlruns ./mlruns
COPY ./notebooks ./notebooks

# 複製主程式入口
COPY . .

# 曝露 FastAPI 端口
EXPOSE 8000

# 啟動 FastAPI
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]