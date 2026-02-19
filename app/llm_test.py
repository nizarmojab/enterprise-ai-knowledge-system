from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

response = llm.invoke("Explique ce qu'est le RAG en 2 phrases simples.")

print(response.content)
