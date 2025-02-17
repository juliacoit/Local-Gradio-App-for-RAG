import json
import ollama
import re
import os
import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader

def load_template():
    template_path = "E:/TCC/Local-Gradio-APP-for-RAG/Local-Gradio-App-for-RAG/json/template.json"
    with open(template_path, "r", encoding="utf-8") as file:
        return json.load(file)

def process_pdf(pdf_file_path):
    if not os.path.exists(pdf_file_path):
        print("Arquivo PDF não encontrado:", pdf_file_path)
        return None
    
    loader = PyMuPDFLoader(pdf_file_path)
    data = loader.load()
    extracted_text = "\n".join([doc.page_content for doc in data])
    
    print(extracted_text[:500])  # Debug: Verifica extração
    return extracted_text

def save_to_file(content, file_name, folder="outputs"):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file_name)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)
    print(f"Arquivo salvo em: {file_path}")
    return file_path

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9À-ÿ\s,;.-]", " ", text)  # Remove caracteres especiais
    text = re.sub(r"\s+", " ", text).strip()  # Remove múltiplos espaços
    return text

def extract_json_from_response(response_text):
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}

def clean_json_response(response_text):
    """
    Extrai e limpa um JSON de uma resposta do modelo.
    - Remove qualquer texto antes/depois do JSON.
    - Garante que o JSON esteja bem formado.
    """
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            json_str = json_str.replace("\n", "").replace("\r", "").strip()  # Removendo quebras de linha desnecessárias
            
            # Valida e tenta carregar o JSON
            json_data = json.loads(json_str)
            return json_data
        else:
            print("Nenhum JSON válido encontrado na resposta!")
            return None
    except json.JSONDecodeError as e:
        print(f"Erro ao carregar JSON: {e}")
        return None
  

def fill_template_with_nuextract(extracted_text, template):
    prompt = f"""Preencha este JSON APENAS com as informações extraídas do texto abaixo.  
**IMPORTANTE:**
- Respeite exatamente a estrutura do JSON fornecido.
- NÃO crie novas chaves ou altere o formato.
- Se não houver informações para um campo, deixe-o vazio, mas NÃO REMOVA a chave.
- Certifique-se de que todas as seções do JSON estejam bem formatadas e completas.

**Template JSON:**
{json.dumps(template, indent=2, ensure_ascii=False)}

**Texto extraído:**  
{clean_text(extracted_text)}

**Responda apenas com o JSON preenchido e nada mais.**
"""

    
    try:
        response = ollama.chat(model="nuextract", messages=[{"role": "user", "content": prompt}])
        response_text = response["message"]["content"]
        print("Raw Model Response:", response_text[:500])  # DEBUG: Ver resposta bruta
        filled_json = clean_json_response(response_text)  
        return filled_json
    except Exception as e:
        print(f"Erro ao interagir com o modelo: {e}")
        return {}

def generate_questions(filled_json):
    prompt = f"""Com base no seguinte JSON estruturado, gere 5 perguntas de múltipla escolha relevantes.
    
    **Conteúdo:**
    {json.dumps(filled_json, indent=2, ensure_ascii=False)}
    
    **Formato esperado:**
    - Pergunta
        - Alternativa A
        - Alternativa B
        - Alternativa C
        - Alternativa D
    """
    
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    response_content = response["message"]["content"]
    final_questions = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()
    
    return final_questions

def process_and_generate(pdf_file):
    pdf_path = pdf_file.name
    file_name_base = re.sub(r"[^\w\-_\.]+", "_", os.path.splitext(os.path.basename(pdf_path))[0])
    
    extracted_text = process_pdf(pdf_path)
    if extracted_text is None:
        return "Falha ao processar o PDF."
    
    save_to_file(extracted_text, f"extracted_text_{file_name_base}.txt")
    
    template = load_template()
    filled_json = fill_template_with_nuextract(extracted_text, template)
    json_content = json.dumps(filled_json, indent=2, ensure_ascii=False)
    save_to_file(json_content, f"filled_json_{file_name_base}.json")
    
    questions = generate_questions(filled_json)
    save_to_file(questions, f"questions_{file_name_base}.txt")
    
    return questions

interface = gr.Interface(
    fn=process_and_generate,
    inputs=gr.File(label="Upload PDF"),
    outputs="text",
    title="Gerador de Perguntas com NuExtract",
    description="Extrai informações do PDF, preenche um modelo JSON e gera perguntas sobre o conteúdo."
)

interface.launch()


