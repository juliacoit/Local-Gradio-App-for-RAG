import gradio as gr

def saudacao(nome):
    return f"Olá, {nome}!"

demo = gr.Interface(fn=saudacao, inputs="text", outputs="text")
demo.launch()