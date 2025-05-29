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
KEYWORDS = {
    "ementa": [
        "ementa", "objetivo", "introdução", "fundamentos", "conceitos", "proposta", "conteúdo programático"
    ],
    "topico": [
        "tópico", "tema", "assunto", "estrutura", "arquitetura", "componente", "modelo", "abordagem", "técnica", "estratégia"
    ],
    "exemplo": [
        "exemplo", "exemplificação", "demonstração", "caso", "código", "aplicação"
    ],
    "subsecao": [
        "detalhe", "subtópico", "aspecto", "variação", "tipo", "subclasse", "submodelo"
    ]
}

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
    
    textos = []
    for i, doc in enumerate(data):
        textos.append(f"--- Página {i + 1} ---\n{doc.page_content.strip()}\n")
    return "\n".join(textos)

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

def structure_text(text, pdf_file_path):

    titulo_pdf = os.path.splitext(os.path.basename(pdf_file_path))[0]

    # Divide o texto por páginas usando o padrão inserido
    paginas = re.split(r"--- Página \d+ ---", text)
    paginas = [p.strip() for p in paginas if p.strip()]  # remove vazios

    structured_data = {
        "titulo": titulo_pdf,
        "disciplina": "",
        "ementa": [],
        "topicos": [],
        "exemplos": []
    }

    def detect_type(line):
        l = line.lower()
        for k, palavras in KEYWORDS.items():
            for p in palavras:
                if p in l:
                    return k
        return None

    for pagina_texto in paginas:
        lines = [l.strip() for l in pagina_texto.split("\n") if l.strip()]
        if not lines:
            continue

        topico = {
            "titulo": lines[0],  # primeira linha da página vira título do tópico
            "palavras-chave": [],
            "conteudo": [],
            "subsecoes": []
        }

        current_subsecao = None

        for line in lines[1:]:  # ignora a primeira linha que já virou título
            tipo = detect_type(line)

            if tipo == "ementa":
                structured_data["ementa"].append(line)

            elif tipo == "exemplo":
                structured_data["exemplos"].append({
                    "descricao": line,
                    "codigo": "",
                    "explicacao": ""
                })

            elif tipo == "subsecao":
                current_subsecao = {
                    "titulo": line,
                    "conteudo": []
                }
                topico["subsecoes"].append(current_subsecao)

            else:
                if current_subsecao:
                    current_subsecao["conteudo"].append(line)
                else:
                    topico["conteudo"].append(line)

        structured_data["topicos"].append(topico)

    # Define o título principal como o do primeiro tópico
    if structured_data["topicos"]:
        structured_data["titulo"] = structured_data["topicos"][0]["titulo"]

    safe_print(f"\n===== Texto estruturado =====\n{json.dumps(structured_data, indent=2, ensure_ascii=False)}\n")
    return structured_data

 #Normaliza e limpa JSON
def clean_json_text(text):
    text = re.sub(r'[\x00-\x1F]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_text(text):
    return unicodedata.normalize("NFC", text)

def generate_prompt(model_name, template, current, text):
    prompt_context = (
        "Você é um assistente que está extraindo informações educacionais de um texto longo, em partes (chunks).\n"
        "A cada etapa, você recebe:\n"
        "- Um modelo JSON com o formato final esperado\n"
        "- Um JSON parcial já preenchido até agora (chamado de \"estado atual\")\n"
        "- Um novo trecho de texto (chunk) com informações adicionais\n\n"
        "Sua tarefa é **completar e atualizar o JSON atual** com base no novo texto fornecido, **mantendo o que já está preenchido**.\n\n"
        "! Retorne **apenas o JSON atualizado**. Não adicione explicações nem comentários.\n"
    )

    if model_name == "gemma3":
        return f"""<|system|>
{prompt_context}
Modelo JSON esperado:
{template}

JSON atual:
{current}

Novo texto:
{text}
<|end|>
<|assistant|>"""

    elif model_name == "llama3":
        return f"""<|start_header_id|>system<|end_header_id|>
{prompt_context}
Modelo JSON esperado:
{template}

JSON atual:
{current}

Novo texto:
{text}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

    elif model_name == "mistral":
        return f"""[INST]
{prompt_context}
Modelo JSON esperado:
{template}

JSON atual:
{current}

Novo texto:
{text}
[/INST]"""

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
<|User|>
Você é um professor criando questões educacionais com base na estrutura da aula abaixo (em formato JSON).

Gere 5 questões de múltipla escolha, cada uma com:
- Enunciado da pergunta
- 4 alternativas (a, b, c, d)
- Indicação clara da alternativa correta
- Uma explicação curta e objetiva para a resposta

! Formato da saída: **texto estruturado, e não JSON**. Siga exatamente o modelo abaixo para cada questão:

---
**1. Pergunta:**
[Enunciado da pergunta]

a) Alternativa A  
b) Alternativa B  
c) Alternativa C  
d) Alternativa D

**Resposta correta:** [Letra]  
**Explicação:** [Explicação da resposta]
---

Aqui está o conteúdo da aula em JSON:

```json
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

    estrutura_inicial = structure_text(texto, pdf_path)
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
            print(f"\n===== Processando chunk =====")
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
        questoes_limpas = re.sub(r'<think>.*?</think>', '', questoes, flags=re.DOTALL)

        path_q = f"resultados/{nome_base}_questoes_{modelo.replace(':', '_')}.txt"
        with open(path_q, "w", encoding="utf-8") as f:
            f.write(questoes_limpas)
        arquivos_questoes[modelo] = path_q

    return (
        jsons_gerados.get("gemma3", ""),
        jsons_gerados.get("llama3", ""),
        jsons_gerados.get("mistral", ""),
        questoes_limpas.get("gemma3", ""),
        questoes_limpas.get("llama3", ""),
        questoes_limpas.get("mistral", ""),
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



