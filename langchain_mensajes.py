from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langchain.schema import AIMessage, HumanMessage

# Definir el estado usando add_messages correctamente
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # Corrección en Annotated

# Crear un estado inicial con mensajes
initial_messages = [
    AIMessage(content="Hola, ¿en qué puedo ayudarte?", name="Model1"),
    HumanMessage(content="¿Cuál es el concepto clave?", name="Sebastian"),
]

# Nuevo mensaje que queremos agregar
new_message = AIMessage(content="El concepto clave es la idea principal de un texto.", name="Model1")

# Usar add_messages de langgraph para agregar el mensaje
updated_state = add_messages(initial_messages, new_message)

# Verificar que el mensaje se ha agregado correctamente
for msg in updated_state:
    print(f"{msg.name}: {msg.content}")
