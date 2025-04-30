# Dockerfile
FROM python:3.11-slim

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Define diretório da aplicação
WORKDIR /app

# Copia arquivos do projeto
COPY . .

# Instala dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta do Gradio
EXPOSE 7860

# Comando para rodar a aplicação
CMD ["python", "app.py"]

ENV OLLAMA_HOST=http://host.docker.internal:11434