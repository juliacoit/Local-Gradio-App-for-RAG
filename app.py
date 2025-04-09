import json
import ollama
import os
import re
import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader
import unicodedata

# Constantes
MAX_INPUT_SIZE = 4000
OVERLAP = 128
TEMPLATE_PATH = "E:/IFBA/TCC/Local-GRadio-App-for_RAG/json/template.json"

list_slms = ["phi3", "llama3", "mistral"]

print("Iniciando Scripts...")

# Função para carregar o template JSON
def load_template():
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

# Função para extrair texto puro do PDF
def extract_text_from_pdf(pdf_file_path):
    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_file_path}")

    loader = PyMuPDFLoader(pdf_file_path)
    data = loader.load()
    return "\n".join([doc.page_content.strip() for doc in data])

# Função para dividir o documento em chunks de forma otimizada
def split_document(document, window_size=MAX_INPUT_SIZE, overlap=OVERLAP):
    words = document.split()
    chunks = []

    if len(words) > window_size:
        for i in range(0, len(words), window_size - overlap):
            chunk = " ".join(words[i:i + window_size])
            chunks.append(chunk)
            if i + len(words[i:i + window_size]) >= len(words):
                break
    else:
        chunks.append(document)

    return chunks

# Função para estruturar qualquer texto extraído
def structure_text(text):
    lines = text.split("\n")
    structured_data = {
        "titulo": "",
        "autor": "",
        "conteudo": []
    }

    current_section = None

    for line in lines:
        line = line.strip()

        # Detecta títulos prováveis (primeiras linhas, letras maiúsculas, tamanho grande)
        if not structured_data["titulo"] and len(line) > 5 and line.istitle():
            structured_data["titulo"] = line
            continue

        # Detecta possíveis autores (normalmente perto do título, contém nome e sobrenome)
        if not structured_data["autor"] and re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+", line):
            structured_data["autor"] = line
            continue

        # Detecta cabeçalhos de seção (geralmente curtos, sem pontuação, em negrito ou maiúsculas)
        if len(line) < 60 and line.isupper():
            current_section = {
                "secao": line,
                "conteudo": []
            }
            structured_data["conteudo"].append(current_section)
            continue

        # Adiciona conteúdo dentro da seção correta
        if current_section:
            current_section["conteudo"].append(line)

    return structured_data

# Função para limpar JSON mal formatado
def clean_json_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove caracteres não ASCII
    text = re.sub(r'\s+', ' ', text)  # Remove múltiplos espaços
    return text.strip()

def normalize_text(text):
    """Normaliza caracteres acentuados malformados no texto."""
    return unicodedata.normalize("NFC", text)

def fix_json(output_text):
    """Corrige erros no JSON gerado pelo modelo."""
    # Remover qualquer coisa após <|end-output|>
    output_text = re.split(r"<\|end-output\|>", output_text)[0].strip()

    # Normalizar caracteres acentuados
    output_text = normalize_text(output_text)

    try:
        # Tentar carregar JSON diretamente
        parsed_json = json.loads(output_text)
        return json.dumps(parsed_json, indent=2, ensure_ascii=False)
    except json.JSONDecodeError as e:
        print("⚠️ Erro ao corrigir JSON:", e)
        return output_text

# Função para enviar chunk para o Ollama com template estruturado
def generate_prompt(model_name, template, current, text):
    if model_name == "phi3":
        return f"""<|system|>
Você é um modelo que extrai informações estruturadas com base em um template.
<|end|>
<|user|>
### Template:
{template}
### Current:
{current}
### Text:
{text}
<|end|>
<|assistant|>"""
    
    elif model_name == "llama3":
        return f"""<|start_header_id|>system<|end_header_id|>
Você é um modelo que extrai informações estruturadas com base em um template.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
### Template:
{template}
### Current:
{current}
### Text:
{text}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

    elif model_name == "mistral":
        return f"""[INST] Você é um modelo que extrai informações estruturadas com base em um template.

### Template:
{template}
### Current:
{current}
### Text:
{text} [/INST]"""
    
    else:
        raise ValueError(f"Modelo {model_name} não suportado.")

def predict_chunk(text, template, current, model_name="phi3"):
    current = clean_json_text(current)
    input_llm = generate_prompt(model_name, template, current, text)

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": input_llm}],
        options={"num_ctx": 4000}
    )

    output_text = response["message"]["content"]

    print("======= RAW OUTPUT FROM OLLAMA =======")
    print(output_text)
    print("======================================")

    output_text_cleaned = output_text.replace("<|end-output|>", "").strip()

    try:
        return json.dumps(json.loads(output_text_cleaned), indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        print("⚠️ WARNING: Invalid JSON output. Returning raw text.")
        return clean_json_text(output_text_cleaned)


# Função principal para processar PDF e enviar ao modelo
def process_and_generate(pdf_file):
    pdf_path = pdf_file.name
    extracted_text = extract_text_from_pdf(pdf_path)

    if not extracted_text:
        return "Falha ao extrair texto do PDF."

    structured_data = structure_text(extracted_text)
    template = json.dumps(load_template(), ensure_ascii=False)
    current_json = json.dumps(structured_data, ensure_ascii=False)
    chunks = split_document(extracted_text)

    # Lista de modelos que você quer testar
    list_slms = ["phi3", "llama3", "mistral"]

    resultados = {}

    for modelo in list_slms:
        print(f"\n========= 🔁 Processando com modelo: {modelo} =========")
        current = current_json
        for i, chunk in enumerate(chunks):
            print(f"🧩 Chunk {i+1}/{len(chunks)} com modelo {modelo}...")
            current = predict_chunk(chunk, template, current, modelo)

        try:
            resultados[modelo] = json.dumps(json.loads(current), indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            resultados[modelo] = current

    return resultados

# Interface Gradio
interface = gr.Interface(
    fn=process_and_generate,
    inputs=gr.File(label="Upload PDF"),
    outputs=gr.JSON(label="Resultados por Modelo"),
    title="Extração de Dados com Múltiplos Modelos via Ollama",
    description="Processa PDFs com diferentes LLMs via Ollama e compara os resultados."
)

print("Iniciando interface do Gradio...")
interface.launch(share=True, debug=True)



