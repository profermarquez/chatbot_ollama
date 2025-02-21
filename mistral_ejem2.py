# ejemplo de uso de ollama con langgraph
from typing import Annotated
from langchain_ollama import OllamaLLM
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langgraph.checkpoint.memory import InMemorySaver  
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

graph_builder = StateGraph(State)

llm = OllamaLLM(
    base_url="http://localhost:11434",
    model="mistral:latest",
    verbose=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

def chatbot(state: State):
    result = llm.invoke(
        "\n".join([message.content for message in state["messages"]])
    )
    from langchain_core.messages import HumanMessage, AIMessage
    return {"messages": [AIMessage(content=result)]}

graph_builder.add_node("chatbot", chatbot)

graph_builder.set_entry_point("chatbot")

# Usar InMemorySaver directamente
memory = InMemorySaver()

graph = graph_builder.compile(checkpointer=memory)

# Ejemplo de uso:
from langchain_core.messages import HumanMessage

initial_message = HumanMessage(content="¿Cuál es el clima en Buenos Aires?")
for output in graph.stream({"messages": [initial_message]}, config={"thread_id": "1", "checkpoint_ns": "ns", "checkpoint_id": "id"}):
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print(value)