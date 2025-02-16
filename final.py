import gradio as gr
import json
import ollama
import re
from langchain_community.document_loaders import PyMuPDFLoader
import os

def load_template():
    # Carregar template JSON predefinido pelo desenvolvedor
    template_path = "E:/TCC/Local-Gradio-APP-for-RAG/Local-Gradio-App-for-RAG/json/template.json"
    with open(template_path, "r", encoding="utf-8") as file:
        template = json.load(file)
    return template

def process_pdf(pdf_bytes):
    if pdf_bytes is None:
        return None
    
    loader = PyMuPDFLoader(pdf_bytes)
    data = loader.load()
    extracted_text = "\n".join([doc.page_content for doc in data])

    print(extracted_text[:500])  # Imprimir os primeiros 500 caracteres para ver se a extração está ok
    return extracted_text

def fill_template_with_nuextract(extracted_text, template):
    prompt = f"""Preencha o seguinte template JSON com base no texto extraído:
    
    Template:
    {json.dumps(template, indent=2, ensure_ascii=False)}
    
    Texto extraído:
    {extracted_text}
    
    JSON preenchido:"""
    
    try:
        response = ollama.chat(
            model="nuextract",
            messages=[{"role": "user", "content": prompt}],
        )
        filled_json = json.loads(response["message"]["content"])
        return filled_json
    except Exception as e:
        print(f"Erro ao interagir com o modelo: {e}")
        return None

def generate_questions(filled_json):
    prompt = f"""Com base no seguinte JSON estruturado, gere 5 perguntas de múltipla escolha relevantes para revisar o conteúdo:

    {json.dumps(filled_json, indent=2, ensure_ascii=False)}
    
    Perguntas:"""
    
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}],
    )
    
    response_content = response["message"]["content"]
    final_questions = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()
    
    return final_questions

def process_and_generate(pdf_bytes):
    template = load_template()
    extracted_text = process_pdf(pdf_bytes)
    if extracted_text is None:
        return "Nenhum PDF enviado."
    
    filled_json = fill_template_with_nuextract(extracted_text, template)
    questions = generate_questions(filled_json)
    
    return questions

interface = gr.Interface(
    fn=process_and_generate,
    inputs=gr.File(label="Upload PDF"),
    outputs="text",
    title="Gerador de Perguntas a partir de PDFs",
    description="Extrai informações do PDF, preenche um modelo JSON e gera perguntas sobre o conteúdo usando Mistral."
)

interface.launch(share=True)