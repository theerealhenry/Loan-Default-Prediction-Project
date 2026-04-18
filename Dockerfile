# =========================================================
# LOAN RISK INTELLIGENCE SYSTEM — PRODUCTION DOCKERFILE
# =========================================================

# Use lightweight Python image
FROM python:3.10-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure logs are flushed immediately
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# =========================================================
# SYSTEM DEPENDENCIES (for LightGBM, SHAP)
# =========================================================

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# =========================================================
# INSTALL PYTHON DEPENDENCIES
# =========================================================

# Copy only requirements first (for caching)
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =========================================================
# COPY PROJECT FILES
# =========================================================

COPY . .

# =========================================================
# STREAMLIT CONFIG
# =========================================================

# Expose Streamlit port
EXPOSE 8501

# Disable telemetry + allow external access
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# =========================================================
# START APPLICATION
# =========================================================

CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]