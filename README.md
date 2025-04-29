🧠 Local RAG App com Gradio + Ollama + PDF

Este projeto é uma aplicação local para extração de dados educacionais estruturados a partir de arquivos PDF, utilizando modelos LLM hospedados localmente via Ollama, com interface feita em Gradio. O sistema também gera automaticamente questões de múltipla escolha com base no conteúdo extraído.
🚀 Funcionalidades

    Extração de texto limpo a partir de arquivos PDF.

    Estruturação inicial do conteúdo (título, autor, seções).

    Uso de modelos LLM locais (phi3:mini, llama3, mistral) para gerar JSON estruturado.

    Correção automática de erros de formatação JSON com json-repair.

    Geração de questões educacionais com o modelo Deepseek-R1.

    Salvamento automático das saídas (arquivos .json e .txt).

🛠️ Requisitos

    Python 3.10+

    Ollama instalado e rodando localmente

    Modelos LLM baixados via Ollama:

        phi3:mini

        llama3

        mistral

        deepseek-r1:8b

    Instalar as dependências Python:

pip install gradio langchain-community pymupdf json-repair

🧱 Estrutura do Projeto

📁 resultados/         ← arquivos gerados (.json e .txt)
📄 app.py              ← script principal com a interface
📄 template.json       ← modelo de estrutura esperado para os dados extraídos

▶️ Como Executar

Inicie o Ollama e certifique-se de que os modelos estão disponíveis localmente:

    ollama run phi3:mini
    ollama run llama3
    ollama run mistral
    ollama run deepseek-r1:8b

Execute o script principal:

    python app.py

A interface será carregada via Gradio. Faça upload de um PDF educacional e aguarde o processamento.

🧠 Modelos Suportados
Modelo	Finalidade
phi3:mini	Extração estruturada de texto para JSON
llama3	Mesma função, com outro modelo LLM
mistral	Alternativa de modelo para extração estruturada
deepseek-r1	Geração de questões com base no conteúdo
🧪 Pipeline de Processamento

Extração de texto: Conversão do PDF para texto puro.

Estruturação inicial: Estimativa de título, autor e seções.

Chunking: Divisão do texto em blocos com sobreposição.

Geração de JSON estruturado: Cada chunk é processado por um LLM via Ollama.

Correção de JSON: Heurísticas + json-repair para garantir estrutura válida.

Geração de questões: Com o modelo Deepseek, baseado no conteúdo extraído.

📂 Saídas

Os arquivos gerados são salvos na pasta resultados/ com nomes no formato:

    NOME_DO_PDF_saida_phi3_mini.json
    NOME_DO_PDF_questoes_phi3_mini.txt

📄 Modelo de JSON Estruturado

O conteúdo extraído é salvo com a seguinte estrutura:

    {
      "titulo": "string",
      "autor": "string",
      "disciplina": "string",
      "data": "string",
      "ementa": ["string"],
      "topicos": [
        {
          "secao": "string",
          "subsecoes": [
            {
              "titulo": "string",
              "conteudo": ["string"]
            }
          ],
          "conteudo": ["string"]
        }
      ],
      "exemplos": [
        {
          "descricao": "string",
          "codigo": "string",
          "explicacao": "string"
        }
      ],
      "referencias": ["string"]
    }

💡 Observações

O template utilizado pelos modelos está em template.json.

Verifique se o caminho para o template.json está correto no código.

A lógica pode ser adaptada para processar outros tipos de conteúdo além do educacional.
