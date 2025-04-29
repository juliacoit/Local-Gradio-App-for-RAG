# 🧠 Local RAG App com Gradio + Ollama + PDF

Este projeto é uma aplicação local para extração de dados educacionais estruturados a partir de arquivos PDF, utilizando modelos LLM hospedados localmente via [Ollama](https://ollama.com/), com interface feita em [Gradio](https://gradio.app/). O sistema também gera automaticamente questões de múltipla escolha com base no conteúdo extraído.

---

## 🚀 Funcionalidades

- Extração de texto limpo a partir de arquivos PDF.
- Estruturação inicial do conteúdo (título, autor, seções).
- Uso de modelos LLM locais (Phi-3, LLaMA 3, Mistral) para gerar JSON estruturado.
- Correção automática de erros de formatação JSON usando `json-repair`.
- Geração de questões educacionais com o modelo **Deepseek-R1**.
- Salvamento automático de saídas (JSON e TXT) por modelo.

---

## 🛠️ Requisitos

- Python 3.10+
- [Ollama](https://ollama.com) instalado e rodando localmente
- Modelos LLM baixados no Ollama (`phi3:mini`, `llama3`, `mistral`, `deepseek-r1:8b`)
- Dependências Python:

```bash
pip install gradio langchain-community pymupdf json-repair
```
🧱 Estrutura do Projeto

📁 resultados/
    └─ arquivos gerados (JSON e TXT)
📄 app.py
📄 template.json  ← modelo de estrutura esperado para os dados extraídos

▶️ Como Executar

    Execute o Ollama e certifique-se de que os modelos estão disponíveis localmente:

ollama run phi3:mini
ollama run llama3
ollama run mistral
ollama run deepseek-r1:8b

    Execute o script principal com:

python app.py

    A interface será carregada via Gradio. Carregue um PDF educacional e aguarde o processamento.

🧠 Modelos Suportados
Nome	Finalidade
phi3:mini	Extração estruturada de texto para JSON
llama3	Mesma função com outro modelo de LLM
mistral	Idem
deepseek	Geração de questões com base no conteúdo
🧪 Pipeline de Processamento

    Extração de texto: Conversão de PDF para texto puro.

    Estruturação inicial: Título, autor e seções são estimados.

    Chunking: Divisão do texto em blocos com sobreposição.

    Geração de JSON estruturado: Cada chunk é processado por um LLM via Ollama.

    Correção de JSON: Aplicação de heurísticas + json-repair.

    Geração de questões: Utilizando modelo Deepseek com base no conteúdo.

📂 Saídas

Os arquivos gerados ficam na pasta resultados/ com nomes no formato:

NOME_DO_PDF_saida_phi3_mini.json
NOME_DO_PDF_questoes_phi3_mini.txt

💡 Observações

    O template usado para guiar os modelos está no arquivo template.json.

    Certifique-se de que o caminho para template.json está correto no código.

    É possível adaptar a lógica para processar outros tipos de conteúdo não educacional.
