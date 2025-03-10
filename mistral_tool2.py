import shutil
import json
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama

messages = []

# Solicitar al usuario el prompt desde la terminal
prompt = input("Introduce tu solicitud: ")

class ObtenerUsoDiscoTool(BaseTool):
    name: str = "obtener_uso_disco"
    description: str = """Obtiene el uso del disco. Llama a esta función siempre que necesites saber el uso del disco, por ejemplo, cuando un cliente pregunte "¿Cuál es el uso del disco?" """

    def _run(self) -> dict:
        path = "/"
        total, usado, libre = shutil.disk_usage(path)
        gb = 1024 * 1024 * 1024

        return {
            "total": f"{total / gb:.2f} GB",
            "usado": f"{usado / gb:.2f} GB",
            "libre": f"{libre / gb:.2f} GB",
        }

    def _arun(self):
        raise NotImplementedError("Does not support async")

class ObtenerHoraEnZonaTool(BaseTool):
    name: str = "obtener_hora_en_zona"
    description: str = """Devuelve la hora actual para una zona horaria dada. Llama a esta función siempre que necesites saber la hora actual de cualquier zona horaria, por ejemplo, cuando un cliente pregunte "¿Cuál es la hora en Katmandú?" """

    def _run(self, zona_horaria: str) -> str:
        try:
            hora_actual = datetime.now(ZoneInfo(zona_horaria))
            return hora_actual.strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception as e:
            return f"Error: Zona horaria no válida: {str(e)}"

    def _arun(self, query: str):
        raise NotImplementedError("Does not support async")

# Registrar herramientas disponibles
lista_herramientas = {
    "obtener_hora_en_zona": ObtenerHoraEnZonaTool(),
    "obtener_uso_disco": ObtenerUsoDiscoTool(),
}
import re
if prompt:
    llm = ChatOllama(model="mistral:latest", base_url="http://localhost:11434", temperature=0)
    llm_con_herramientas = llm.bind_tools(list(lista_herramientas.values()))

    messages.append(HumanMessage(prompt))
    respuesta_ai = llm_con_herramientas.invoke(messages)

    try:
        tool_call_info = json.loads(respuesta_ai.content)
        if isinstance(tool_call_info, list) and tool_call_info:
            tool_call = tool_call_info[0]
            tool_name = tool_call["name"]
            tool_arguments = tool_call.get("arguments", {})

            print(f"Herramienta detectada: {tool_name}")

            if tool_name in lista_herramientas:
                herramienta = lista_herramientas[tool_name]
                if isinstance(tool_arguments, dict):
                    resultado = herramienta._run(**tool_arguments)
                else:
                    resultado = herramienta._run()

                #invoco de nuevo pero sin tools
                llm_sin_herramientas = ChatOllama(model="mistral:latest", base_url="http://localhost:11434", temperature=0).bind_tools([])

                resultado= llm_sin_herramientas.invoke([HumanMessage("Responde de manera directa sin herramientas."), HumanMessage(prompt)])

                #print("Resultado:", resultado.content)
                resultado_limpio = re.sub(r'^\[.*?\]\n', '', resultado.content, flags=re.DOTALL)
                resultado_limpio=resultado_limpio.strip()
                print("Resultado limpio:", resultado_limpio)
            else:
                print("Herramienta no encontrada.")
            
    except json.JSONDecodeError:
        print(respuesta_ai.content)
    except Exception as e:
        print(f"Ocurrió un error: {e}")
