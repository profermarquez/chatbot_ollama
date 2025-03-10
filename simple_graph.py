from typing import TypedDict, Literal
import random
from langgraph.graph import StateGraph, START, END
import graphviz

class State(TypedDict):
    graph_state: str

def node1(state: State):
    print("Node 1")
    return {"graph_state": state["graph_state"] + " -> i am node1"}

def node2(state: State):
    print("Node 2")
    return {"graph_state": state["graph_state"] + " -> i am node2"}

def node3(state: State):
    print("Node 3")
    return {"graph_state": state["graph_state"] + " -> i am node3"}

# Función de decisión
def decide_mood(state: State) -> Literal["node2", "node3"]:
    return "node2" if random.random() < 0.5 else "node3"

# Construcción del grafo
builder = StateGraph(State)
builder.add_node("node1", node1)
builder.add_node("node2", node2)
builder.add_node("node3", node3)

# Agregar transiciones
builder.add_edge(START, "node1")
builder.add_conditional_edges("node1", decide_mood)
builder.add_edge("node2", END)
builder.add_edge("node3", END)

# Compilar el grafo
graph = builder.compile()

# Generar el gráfico con graphviz
dot = graphviz.Digraph(comment="LangGraph")

# Agregar nodos
for node_name in graph.nodes:
    dot.node(node_name, node_name)

# ✅ Agregar aristas correctamente
for edge in graph.get_graph().edges:
    start_node = edge.source
    end_node = edge.target
    dot.edge(start_node, end_node)

# Guardar y mostrar el gráfico
dot.render("langgraph_graph", view=True, format="png")