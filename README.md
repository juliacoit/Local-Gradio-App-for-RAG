# 📄 PDF Educational Data Extractor with LLMs (Phi3, LLaMA3, Mistral) + DeepSeek MCQ Generator

Este projeto é uma aplicação em Python com interface Gradio que permite **processar PDFs educacionais**, extrair e estruturar dados com modelos LLM locais via **Ollama**, e **gerar questões de múltipla escolha (MCQs)** com o modelo DeepSeek.

---

## 🚀 Funcionalidades

- 📥 Upload de arquivos PDF
- 🧠 Extração e estruturação de dados com:
  - 🔹 Phi-3
  - 🔹 LLaMA 3
  - 🔹 Mistral
- 💾 Salvamento de arquivos JSON com as saídas dos modelos
- 📝 Geração automática de questões de múltipla escolha com o modelo DeepSeek
- ⬇️ Download dos arquivos JSON e TXT processados

---

## 📊 Fluxo do Sistema

```mermaid
flowchart LR
    A[Upload do PDF] --> B[Extração de texto]
    B --> C[Pré-processamento]
    C --> D{Envio para modelos}

    D --> D1[Phi3 + Prompt]
    D --> D2[LLaMA3 + Prompt]
    D --> D3[Mistral + Prompt]

    D1 --> E[Correção JSON]
    D2 --> E
    D3 --> E

    E --> F[Salvar arquivos JSON]
    F --> G[Geração de questões]
    G --> H[Download dos arquivos]

🛠️ Requisitos

    Python 3.10+

    Ollama

    Modelos:

        phi3:mini

        llama3

        mistral

        deepseek-coder ou similar para MCQ

    Dependências:

        gradio

        PyMuPDF (fitz)

        json

        os, uuid, datetime

        langchain, chromadb (opcional)

▶️ Como Executar

    Clone o repositório:

git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio

Instale as dependências:

pip install -r requirements.txt

Certifique-se de que o Ollama esteja rodando com os modelos baixados:

ollama run phi3
ollama run llama3
ollama run mistral

Execute o aplicativo:

    python app.py

📁 Estrutura do Projeto

.
├── app.py                # Interface principal com Gradio
├── indexing.py           # Indexação vetorial com ChromaDB (opcional)
├── prompts/              # Prompts usados por cada modelo
├── outputs/
│   ├── json/             # Arquivos JSON gerados pelos modelos
│   └── txt/              # Arquivos de texto extraídos
├── README.md
└── requirements.txt

📌 Observações

    Você pode adaptar os prompts individualmente para cada modelo no diretório prompts/.

    Os arquivos são salvos com nomes únicos para facilitar a organização e o download.

    O DeepSeek pode ser substituído por qualquer outro modelo de geração de perguntas, desde que adaptado no código.
