## Por enquanto estou tentando fazer o modelo extrair o JSON corretamente apenas, sem a elaboração de questões

import json
import os
import re
import ollama
import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader
import unicodedata

MAX_INPUT_SIZE = 4000
MAX_NEW_TOKENS = 6000
OVERLAP = 128

def load_template():
    template_path = "E:/TCC/Local-Gradio-APP-for-RAG/Local-Gradio-App-for-RAG/json/template.json"
    with open(template_path, "r", encoding="utf-8") as file:
        return json.load(file)

def process_pdf(pdf_file_path):
    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_file_path}")
    
    loader = PyMuPDFLoader(pdf_file_path)
    data = loader.load()
    
    extracted_data = {
        "titulo": "",
        "autor": "",
        "ementa": [],
        "conteudo": [],
        "referencias": []
    }
    
    for doc in data:
        page_number = doc.metadata.get("page", 0)  # Garante que page_number tenha um valor padrão
        text = doc.page_content.strip()

        # Verifica título, autor e ementa na primeira página
        if page_number == 1:
            lines = text.split("\n")
            if len(lines) > 2:
                extracted_data["titulo"] = lines[0].strip()
                extracted_data["autor"] = lines[1].strip()
                extracted_data["ementa"] = [line.strip() for line in lines[2:5] if line.strip()]
        
        # Estruturar conteúdo por página
        topicos = [line.strip() for line in text.split("\n") if line.strip()]  # Remove linhas vazias
        
        extracted_data["conteudo"].append({
            "página": page_number,
            "secao": "",  # Poderia ser extraído automaticamente caso existam seções no texto
            "topicos": topicos
        })
        
        # Identificar referências bibliográficas
        if "Referências" in text:
            extracted_data["referencias"].extend(topicos)  # Adiciona tudo como referência
    
    return extracted_data


def save_to_file(content, file_name, folder="outputs"):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file_name)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(json.dumps(content, indent=2, ensure_ascii=False))
    print(f"Arquivo salvo em: {file_path}")
    return file_path

def split_document(document, window_size=MAX_INPUT_SIZE, overlap=OVERLAP):
    # Apenas dividir o documento sem tokenização explícita
    chunks = []
    if len(document) > window_size:
        for i in range(0, len(document), window_size - overlap):
            chunk = document[i:i + window_size]
            chunks.append(chunk)
            if i + len(document[i:i + window_size]) >= len(document):
                break
    else:
        chunks.append(document)
    print(f"\tSplit into {len(chunks)} chunks")
    return chunks

def clean_text(text):
    # Remover caracteres não-ASCII e inválidos
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[\ufffd\u219f\u014d\u03b2\u2026]', ' ', text)
    text = re.sub(r'\s+', ' ', text)  # Substituir múltiplos espaços por um único
    return text.strip()

def predict_nuextract(text, template, model_name="nuextract"):
    template_json = json.dumps(template, indent=4)
    chunks = split_document(text, MAX_INPUT_SIZE, OVERLAP)
    
    prev = template_json
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i}...")
        prompt = f"""
        <|input|>
        ### Template:
        {prev}
        ### Text:
        {chunk}
        
        <|output|>
        """
        
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        output_text = response["message"]["content"].strip()

        print(f"Output text from Ollama: {output_text}")  # Para depuração

        if not output_text:
            print("Resposta vazia do modelo.")
            continue
        
        # Pegando o texto antes de <|end-output|>
        if "<|end-output|>" in output_text:
            output_text = output_text.split("<|end-output|>")[0].strip()

        # Tenta limpar e converter JSON
        try:
            cleaned_text = clean_text(output_text)
            parsed_json = json.loads(cleaned_text)

            # Se JSON estiver vazio ou inválido, mantém a versão anterior
            if not parsed_json or all([(v in ["", []]) for v in parsed_json.values()]):
                print("JSON vazio ou inválido, mantendo a versão anterior.")
                continue
            
            prev = parsed_json  # Atualiza JSON processado
        except json.JSONDecodeError as e:
            print(f"Erro ao decodificar JSON extraído: {e}")
            continue
    
    return prev


def process_and_generate(pdf_file):
    pdf_path = pdf_file.name
    file_name_base = re.sub(r"[^\w\-_\.]+", "_", os.path.splitext(os.path.basename(pdf_path))[0])
    
    extracted_data = process_pdf(pdf_path)
    if extracted_data is None:
        return "Falha ao processar o PDF."
    
    save_to_file(extracted_data, f"extracted_json_{file_name_base}.json")
    
    template = load_template()
    filled_json = predict_nuextract(json.dumps(extracted_data, ensure_ascii=False), template)
    
    save_to_file(filled_json, f"filled_json_{file_name_base}.json")
    
    return json.dumps(filled_json, indent=2, ensure_ascii=False)


interface = gr.Interface(
    fn=process_and_generate,
    inputs=gr.File(label="Upload PDF"),
    outputs="text",
    title="Gerador de JSON com NuExtract e Ollama",
    description="Extrai informações estruturadas do PDF e preenche um modelo JSON usando o NuExtract via Ollama."
)

interface.launch()


