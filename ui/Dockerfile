FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "ui/Home.py", "--server.address=0.0.0.0", "--server.port=8501"]