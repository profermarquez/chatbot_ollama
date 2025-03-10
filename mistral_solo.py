# ejemplo de uso de ollama con langgraph
from langchain_ollama import OllamaLLM
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage


llm = OllamaLLM(
    base_url="http://localhost:11434",
    model="mistral:latest",
    verbose=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

initial_message = HumanMessage(content="¿Cuál es el clima en Buenos Aires?")
response = llm.invoke(initial_message.content)

ai_message = AIMessage(content=response)
print("\nRespuesta del modelo:")
print(ai_message.content)