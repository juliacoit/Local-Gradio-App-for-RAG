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

list_slms = ["phi3:mini", "llama3", "mistral"]

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

        # Identificar possível autor com base em "Prof." ou "Professor"
        if not structured_data["autor"] and re.search(r"(Prof\.|Professor[a]?)", line, re.IGNORECASE):
            structured_data["autor"] = line
            continue

        # Guardar possíveis títulos nas primeiras linhas
        if i < 5 and 10 < len(line) < 100 and not re.search(r"(prof\.|instituto|ifba|ifma|ifrn|campus|colegio|escola)", line, re.IGNORECASE):
            candidate_titles.append(line)

        # Detectar seção por linha em maiúsculas curtas
        if len(line) < 60 and line.isupper():
            current_section = {
                "secao": line,
                "conteudo": []
            }
            structured_data["conteudo"].append(current_section)
            continue

        # Detectar linha com marcador de tópico (•)
        if line.startswith("•") or line.startswith("- "):
            if not current_section:
                current_section = {
                    "secao": "Tópicos",
                    "conteudo": []
                }
                structured_data["conteudo"].append(current_section)
            current_section["conteudo"].append(line)
            continue

        # Conteúdo normal dentro da seção
        if current_section:
            current_section["conteudo"].append(line)

    # Seleciona o título mais provável (o mais longo entre os candidatos)
    if candidate_titles and not structured_data["titulo"]:
        structured_data["titulo"] = max(candidate_titles, key=len)

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
    """Corrige erros comuns e valida JSON, retornando como string formatada ou None."""
    # Remover qualquer coisa após <|end-output|>
    output_text = re.split(r"<\|end-output\|>", output_text)[0].strip()
    output_text = normalize_text(output_text)

    # Tentativa direta
    try:
        parsed = json.loads(output_text)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except:
        pass

    # Correções comuns: remover vírgulas a mais antes de } ou ]
    output_text = re.sub(r",\s*([}\]])", r"\1", output_text)

    try:
        parsed = json.loads(output_text)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except Exception as e:
        print("❌ JSON ainda inválido após tentativas de correção:", e)
        return None  # agora retornamos None explicitamente


# Função para enviar chunk para o Ollama com template estruturado
def generate_prompt(model_name, template, current, text):
    if model_name == "phi3:mini":
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

def predict_chunk(text, template, current, model_name="phi3:mini"):
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
        print("WARNING: Invalid JSON output. Returning raw text.")
        return clean_json_text(output_text_cleaned)

def gerar_questoes(json_path, modelo_deepseek="deepseek-r1:8b"):
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    topicos = ""
    for sec in json_data.get("conteudo", []):
        nome_secao = sec.get("secao", "Seção")
        conteudo = ", ".join(sec.get("conteudo", []))
        topicos += f"- {nome_secao}: {conteudo}\n"

    prompt = f"""
<｜User｜>Você é um professor criando questões educacionais com base nos seguintes tópicos:

{topicos}

Gere 5 questões de múltipla escolha com:
- 4 alternativas (a, b, c, d)
- Marque a resposta correta
- Dê uma explicação para a resposta

Formato:
1. Pergunta?
   a) ...
   b) ...
   c) ...
   d) ...
   Resposta correta: ...
   ℹExplicação: ...<｜Assistant｜>"""

    resposta = ollama.chat(
        model=modelo_deepseek,
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
        json_final = fix_json(current)
        if not json_final:
            print(f"⚠️ JSON inválido mesmo após fix_json para modelo {modelo}")
            json_final = json.dumps({"erro": "JSON inválido mesmo após correção"}, indent=2, ensure_ascii=False)


        jsons_gerados[modelo] = json_final

        # Salvar JSON
        path_json = f"resultados/{nome_base}_saida_{modelo.replace(':', '_')}.json"
        with open(path_json, "w", encoding="utf-8") as f:
            f.write(json_final)
        arquivos_json[modelo] = path_json

        # Gerar questões
        try:
            questoes = gerar_questoes(path_json)
        except Exception as e:
            questoes = f"Erro ao gerar questões com Deepseek: {e}"

        questoes_geradas[modelo] = questoes

        # Salvar questões
        path_q = f"resultados/{nome_base}_questoes_{modelo.replace(':', '_')}.txt"
        with open(path_q, "w", encoding="utf-8") as f:
            f.write(questoes)
        arquivos_questoes[modelo] = path_q

    return (
        jsons_gerados.get("phi3:mini", ""),
        jsons_gerados.get("llama3", ""),
        jsons_gerados.get("mistral", ""),
        questoes_geradas.get("phi3:mini", ""),
        questoes_geradas.get("llama3", ""),
        questoes_geradas.get("mistral", ""),
        arquivos_json.get("phi3:mini", ""),
        arquivos_json.get("llama3", ""),
        arquivos_json.get("mistral", ""),
        arquivos_questoes.get("phi3:mini", ""),
        arquivos_questoes.get("llama3", ""),
        arquivos_questoes.get("mistral", "")
    )

# Gradio Interface
interface = gr.Interface(
    fn=processar_tudo,
    inputs=gr.File(label="Upload do PDF"),
    outputs=[
        gr.JSON(label="JSON - phi3"),
        gr.JSON(label="JSON - llama3"),
        gr.JSON(label="JSON - mistral"),
        gr.Textbox(label="Questões - phi3", lines=10),
        gr.Textbox(label="Questões - llama3", lines=10),
        gr.Textbox(label="Questões - mistral", lines=10),
        gr.File(label="Download JSON - phi3"),
        gr.File(label="Download JSON - llama3"),
        gr.File(label="Download JSON - mistral"),
        gr.File(label="Download Questões - phi3"),
        gr.File(label="Download Questões - llama3"),
        gr.File(label="Download Questões - mistral")
    ],
    title="Extração + Questões Educacionais com 3 Modelos + Deepseek",
    description="Faça upload de um PDF, gere JSONs estruturados com phi3, llama3 e mistral e compare as questões geradas com Deepseek."
)

if __name__ == "__main__":
    print("Iniciando interface do Gradio...")
    interface.launch(share=True, debug=True)



