import json
import ollama
import os
import re
import gradio as gr
import sys
import io
from langchain_community.document_loaders import PyMuPDFLoader
import unicodedata
from json_repair import repair_json

# Constantes
MAX_INPUT_SIZE = 8000
OVERLAP = 200
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(BASE_DIR, "json", "template.json")

list_slms = ["gemma3", "llama3", "mistral"]

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
    candidate_titles = []

    for i, line in enumerate(lines):
        line = line.strip()

        if not structured_data["autor"] and re.search(r"(Prof\\.|Professor[a]?)", line, re.IGNORECASE):
            structured_data["autor"] = line
            continue

        if i < 5 and 10 < len(line) < 100 and not re.search(r"(prof\\.|instituto|ifba|ifma|ifrn|campus|colegio|escola)", line, re.IGNORECASE):
            candidate_titles.append(line)

        if len(line) < 60 and line.isupper():
            current_section = {"secao": line, "conteudo": []}
            structured_data["conteudo"].append(current_section)
            continue

        if line.startswith("\u2022") or line.startswith("- "):
            if not current_section:
                current_section = {"secao": "Tópicos", "conteudo": []}
                structured_data["conteudo"].append(current_section)
            current_section["conteudo"].append(line)
            continue

        if current_section:
            current_section["conteudo"].append(line)

    if candidate_titles and not structured_data["titulo"]:
        structured_data["titulo"] = max(candidate_titles, key=len)

    return structured_data

# Normaliza e limpa JSON
def clean_json_text(text):
    text = re.sub(r'[\x00-\x1F]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_text(text):
    return unicodedata.normalize("NFC", text)

def generate_prompt(model_name, template, current, text):
    if model_name == "gemma3":
        return f"""<|system|>
Leia o texto abaixo e retorne os dados extraídos no seguinte formato JSON, sem explicações ou comentários:
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
Leia o texto abaixo e retorne os dados extraídos no seguinte formato JSON, sem explicações ou comentários:<|eot_id|>
<|start_header_id|>user<|end_header_id|>
### Template:
{template}
### Current:
{current}
### Text:
{text}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    elif model_name == "mistral":
        return f"""[INST] Leia o texto abaixo e retorne os dados extraídos no seguinte formato JSON, sem explicações ou comentários:

### Template:
{template}
### Current:
{current}
### Text:
{text} [/INST]"""
    else:
        raise ValueError(f"Modelo {model_name} não suportado.")

# Corrige o JSON gerado
def fix_json(output_text):
    """Tenta corrigir JSON com vírgulas finais inválidas e normaliza o texto, usando também json-repair."""
    output_text = re.split(r"<\|end-output\|>", output_text)[0].strip()
    output_text = unicodedata.normalize("NFC", output_text)

    # Captura apenas o primeiro JSON válido
    try:
        # Tenta localizar o primeiro JSON com pilha de chaves
        stack = []
        start = None
        for i, char in enumerate(output_text):
            if char == '{':
                if start is None:
                    start = i
                stack.append('{')
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack:
                        json_chunk = output_text[start:i+1]
                        break
        else:
            json_chunk = output_text  # fallback
    except Exception as e:
        print("Erro ao isolar JSON:", e)
        json_chunk = output_text

    # Remove vírgulas finais inválidas
    json_chunk = re.sub(r",\s*([}\]])", r"\1", json_chunk)

    # Tenta carregar com json normal
    try:
        parsed = json.loads(json_chunk)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except Exception as e:
        print("JSON inválido, tentando corrigir com json-repair...")
        try:
            repaired = repair_json(json_chunk)
            parsed = json.loads(repaired)
            print("JSON corrigido com json-repair.")
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except Exception as e2:
            print("Falha ao corrigir com json-repair também:", str(e2).encode("utf-8", errors="replace"))
            return json_chunk  # retorna o texto bruto se não conseguir corrigir

def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        # Se houver erro de codificação, converta para UTF-8 e imprima
        print(text.encode('ascii', errors='replace').decode('ascii'))

def predict_chunk(text, template, current, model_name="gemma3"):
    current = clean_json_text(current)
    input_llm = generate_prompt(model_name, template, current, text)

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": input_llm}],
        options={"num_ctx": 4000}
    )

    output_text = response["message"]["content"]
    safe_print(f"\n===== RAW OUTPUT from {model_name} =====\n{output_text}\n")
    output_text_cleaned = output_text.replace("<|end-output|>", "").strip()

    # Corrigir o JSON antes de retornar
    json_corrigido = fix_json(output_text_cleaned)
    safe_print(f"\n===== JSON de  {model_name} CORRIGIDO =====\n{json_corrigido}\n")

    return json_corrigido

def gerar_questoes(json_corrigido, modelo_questoes="qwen3:8b"):
    import json

    with open(json_corrigido, "r", encoding="utf-8") as f:
        json_data = json.load(f)



    prompt = f"""
<｜User｜>Você é um professor criando questões educacionais com base na estrutura de aula abaixo (em formato JSON). 

Gere 5 questões de múltipla escolha com:
- 4 alternativas (a, b, c, d)
- Marque a resposta correta
- Dê uma explicação para a resposta

JSON da aula:

json
{json.dumps(json_data, indent=2, ensure_ascii=False)}

<｜Assistant｜>"""

    resposta = ollama.chat(
        model=modelo_questoes,
        messages=[{"role": "user", "content": prompt}]
    )

    return resposta["message"]["content"]


def processar_tudo(pdf_file):
    pdf_path = pdf_file.name
    nome_base = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs("resultados", exist_ok=True)

    texto = extract_text_from_pdf(pdf_path)
    with open(f"resultados/{nome_base}_texto.txt", "w", encoding="utf-8") as f:
        f.write(texto)

    estrutura_inicial = structure_text(texto)
    current_json = json.dumps(estrutura_inicial, ensure_ascii=False)
    template = json.dumps(load_template(), ensure_ascii=False)
    chunks = split_document(texto)

    jsons_gerados = {}
    questoes_geradas = {}
    arquivos_json = {}
    arquivos_questoes = {}

    for modelo in list_slms:
        print(f"\n========= Processando com modelo: {modelo} =========")
        current = current_json
        for chunk in chunks:
            current = predict_chunk(chunk, template, current, modelo)

        try:
            json_final = fix_json(current)
            if json_final is None:
                json_final = current  # salva bruto se ainda inválido
        except Exception as e:
            print("Erro ao aplicar fix_json:", e)
            json_final = current

        jsons_gerados[modelo] = json_final

        # Salvar JSON
        path_json = f"resultados/{nome_base}_saida_{modelo.replace(':', '_')}.json"
        with open(path_json, "w", encoding="utf-8") as f:
            f.write(json_final)
        arquivos_json[modelo] = path_json

        try:
            questoes = gerar_questoes(path_json)
        except Exception as e:
            questoes = f"Erro ao gerar questões: {e}"

        questoes_geradas[modelo] = questoes

        path_q = f"resultados/{nome_base}_questoes_{modelo.replace(':', '_')}.txt"
        with open(path_q, "w", encoding="utf-8") as f:
            f.write(questoes)
        arquivos_questoes[modelo] = path_q

    return (
        jsons_gerados.get("gemma3", ""),
        jsons_gerados.get("llama3", ""),
        jsons_gerados.get("mistral", ""),
        questoes_geradas.get("gemma3", ""),
        questoes_geradas.get("llama3", ""),
        questoes_geradas.get("mistral", ""),
        arquivos_json.get("gemma3", ""),
        arquivos_json.get("llama3", ""),
        arquivos_json.get("mistral", ""),
        arquivos_questoes.get("gemma3", ""),
        arquivos_questoes.get("llama3", ""),
        arquivos_questoes.get("mistral", "")
    )

# Gradio Interface
interface = gr.Interface(
    fn=processar_tudo,
    inputs=gr.File(label="Upload do PDF"),
    outputs=[
        gr.JSON(label="JSON - gemma3"),
        gr.JSON(label="JSON - llama3"),
        gr.JSON(label="JSON - mistral"),
        gr.Textbox(label="Questões - gemma3", lines=10),
        gr.Textbox(label="Questões - llama3", lines=10),
        gr.Textbox(label="Questões - mistral", lines=10),
        gr.File(label="Download JSON - gemma3"),
        gr.File(label="Download JSON - llama3"),
        gr.File(label="Download JSON - mistral"),
        gr.File(label="Download Questões - gemma3"),
        gr.File(label="Download Questões - llama3"),
        gr.File(label="Download Questões - mistral")
    ],
    title="Extração + Questões Educacionais com 3 Modelos + Qwen3",
    description="Faça upload de um PDF, gere JSONs estruturados com phi3, llama3 e mistral e compare as questões geradas com Deepseek."
)

if __name__ == "__main__":
    print("Iniciando interface do Gradio...")
    interface.launch(share=True, debug=True)



