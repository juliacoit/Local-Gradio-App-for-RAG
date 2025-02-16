# Gerador de Perguntas a partir de PDFs com Gradio e Modelos de IA

Este projeto utiliza o Gradio para criar uma interface de upload de PDFs, extrair o conteúdo, estruturar os dados em um JSON e, com o auxílio dos modelos *nuextract* e *mistral*, gerar perguntas baseadas no conteúdo extraído.

## **Visão Geral**

1. **Upload de um PDF**: O usuário carrega o arquivo.
2. **Extração de Texto**: O texto é extraído usando o `PyMuPDFLoader`.
3. **Preenchimento do Template JSON**: O modelo *nuextract* organiza as informações no formato JSON.
4. **Geração de Perguntas**: O modelo *mistral* cria perguntas sobre o conteúdo.
5. **Exibição das Perguntas**: As perguntas geradas são mostradas na interface Gradio.

## **Estrutura de Arquivos**

```
Local-Gradio-App-for-RAG/
    ├── final.py               # Código principal
    ├── json/
    │   └── template.json      # Estrutura inicial para o preenchimento
    └── README.md              # Documento atual
```

### **template.json**
```json
{
    "titulo": "",
    "autor": "",
    "ementa": [],
    "conteudo": [
      {
        "página": "",
        "secao": "",
        "topicos": []
      }
    ],
    "referencias": []
}
```

## **Explicação do Código**

### 1. **Bibliotecas Utilizadas**
- `gradio`: Interface gráfica.
- `json`: Manipulação de arquivos JSON.
- `ollama`: Comunicação com os modelos de IA.
- `re`: Limpeza das respostas geradas.
- `langchain_community.document_loaders`: Extração de texto do PDF.
- `os`: Operações de sistema.

### 2. **Funções Principais**

#### 2.1. `load_template()`
Carrega o template JSON predefinido. Se estiver vazio, retorna um dicionário vazio.

#### 2.2. `process_pdf(pdf_bytes)`
Recebe o arquivo PDF, extrai o texto de todas as páginas e imprime os primeiros 500 caracteres para depuração.

#### 2.3. `fill_template_with_nuextract(extracted_text, template)`
Envia o texto extraído junto ao template para o modelo *nuextract*, que estrutura as informações no JSON.

#### 2.4. `generate_questions(filled_json)`
Utiliza o modelo *mistral* para criar perguntas baseadas no conteúdo estruturado.

#### 2.5. `process_and_generate(pdf_bytes)`
Fluxo principal que integra as etapas anteriores.

### **Exemplo de Execução**

1. Certifique-se de ter o Python e as bibliotecas necessárias instaladas.
2. Execute o comando:

```bash
python final.py
```

3. Acesse a interface no navegador pelo endereço:

```
http://127.0.0.1:7860
```

4. Faça o upload de um PDF e aguarde as perguntas geradas.

## **Possíveis Problemas e Soluções**

### **1. Erro de Decodificação JSON**
- Certifique-se de que o `template.json` não está vazio.
- Verifique o caminho do arquivo.

### **2. Link de Compartilhamento Não Gerado**
- Verifique sua conexão com a internet.
- Execute o Gradio com `share=False` caso não precise de um link público.

### **3. Respostas Incompletas**
- Verifique a estrutura do PDF.
- Ajuste o prompt enviado ao modelo para maior clareza.

---
**Desenvolvido com Python, Gradio e IA para facilitar a criação de questões educacionais.**

