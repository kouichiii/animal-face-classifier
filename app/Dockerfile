# ── my-catdog-app/Dockerfile ──
FROM python:3.11-slim

# ① 依存ライブラリ
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ② アプリ本体 & モデル
COPY . .

# ③ Cloud Run のデフォルト 8080 ポートを使用
ENV PORT=8080
EXPOSE 8080

# ④ Streamlit を headless モードで起動
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
