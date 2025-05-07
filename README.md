# 📄 PDF Educational Data Extractor with LLMs (Gemma3, LLaMA3, Mistral) + Qwen3 MCQ Generator

Este projeto é uma aplicação em Python com interface Gradio que permite **processar PDFs educacionais**, extrair e estruturar dados com modelos LLM locais via **Ollama**, e **gerar questões de múltipla escolha (MCQs)** com o modelo Qwen3.

---

## 🚀 Funcionalidades

- 📥 Upload de arquivos PDF
- 🧠 Extração e estruturação de dados com:
  - 🔹 Gemma3
  - 🔹 LLaMA 3
  - 🔹 Mistral
- 💾 Salvamento de arquivos JSON com as saídas dos modelos
- 📝 Geração automática de questões de múltipla escolha com o modelo Qwen3
- ⬇️ Download dos arquivos JSON e TXT processados

-
🛠️ Requisitos

    Python 3.10+

    Ollama

    Modelos:

        Gemma3

        llama3

        mistral

        Qwen3 ou similar para MCQ

    Dependências:

        gradio

        PyMuPDF (fitz)

        json

        os, uuid, datetime

        langchain

▶️ Como Executar

  Clone o repositório:

    git clone https://github.com/juliacoit/Local-Gradio-App-for-RAG
    cd Local-Gradio-App-for-RAG

Instale as dependências:

    pip install -r requirements.txt

Certifique-se de que o Ollama esteja rodando com os modelos baixados:

    ollama run gamma3
    ollama run llama3
    ollama run mistral

Inicie o Ollama e Execute o aplicativo:

    ollama serve
    python app.py

📁 Estrutura do Projeto

.
├── app.py                # Interface principal com Gradio
├── outputs/
│   ├── json/             # Arquivos JSON gerados pelos modelos
│   └── txt/              # Arquivos de texto extraídos
├── README.md
└── requirements.txt

📌 Observações

Você pode adaptar os prompts individualmente para cada modelo no diretório prompts/.

Os arquivos são salvos com nomes únicos para facilitar a organização e o download.
