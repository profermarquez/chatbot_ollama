from random import random
from langchain_core.tools import tool
from langchain_ollama import ChatOllama



@tool
def devolver_random()-> any:
    """
    Genera un número aleatorio entero entre 1 y 100.
    Genera un numero aleatoreo
    """
    result = random.randint(1, 100)
    print(result)


llm = ChatOllama(
    base_url="http://localhost:11434",
    model="mistral:latest",
    temperature=0,
).bind_tools([devolver_random])


result = llm.invoke(
    "Cual es la capital de Ukrania? "
    "Misiones se encuentra en Paraguay? "
    "devolver_random"
)
print(result)
            