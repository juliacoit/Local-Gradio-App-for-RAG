import ollama

response = ollama.chat(
    model="phi3:mini",
    messages=[{"role": "user", "content": "Explique o que é aprendizado supervisionado."}]
)

print(response["message"]["content"])
