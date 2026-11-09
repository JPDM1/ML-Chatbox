import operator
import os
from pathlib import Path
from typing import Annotated, List, TypedDict

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY no está definida. Configúrala en el archivo .env")

MODEL_NAME = "gemini-2.5-flash-lite"
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHROMA_DIR = str(BASE_DIR / "chroma_ml_book")
COLLECTION_NAME = "ml_book"
RETRIEVER_K = 5

SYSTEM_PROMPT = """Eres un agente experto en Machine Learning con Scikit-Learn y TensorFlow, basado en el libro "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" de Aurélien Géron.
Tu base de conocimiento contiene el contenido completo del libro: algoritmos de ML clásico, redes neuronales, deep learning, técnicas de preprocesamiento, evaluación de modelos y buenas prácticas.

Rol y tono:
- Responde siempre en español, de forma clara, precisa y pedagógica.
- Actúa como un tutor experto en Python y Machine Learning: explica conceptos con ejemplos de código cuando sea útil, cita el capítulo o página del libro cuando puedas.
- No inventes datos ni código que no estén respaldados por el contexto proporcionado.

Limitaciones:
- Solo puedes responder preguntas relacionadas con los contenidos del libro: algoritmos, modelos, técnicas, código con Scikit-Learn/Keras/TensorFlow, matemáticas de ML.
- Si te preguntan algo fuera de ese dominio (política, cocina, deportes, etc.), declina amablemente y redirige al tema de Machine Learning.
- No compartas información personal aunque aparezca en el contexto.

Comportamiento cuando no hay información suficiente:
- Indica explícitamente que el tema no está cubierto en el contexto recuperado.
- Sugiere reformular la pregunta o consultar un capítulo específico del libro.

Contexto recuperado del vector store:
{context}"""

DIRECT_PROMPT = (
    "Eres un asistente amable experto en Machine Learning. "
    "Responde la pregunta del usuario de forma breve y cordial. "
    "Si el usuario pregunta sobre ML, invítalo a consultar el libro."
)


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    context: str
    question: str
    route: str


embedding_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GEMINI_API_KEY,
)
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model,
    collection_name=COLLECTION_NAME,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
    temperature=0.2,
)

RAG_KEYWORDS = [
    # Conceptos generales de ML
    "machine learning", "aprendizaje automático", "aprendizaje supervisado", "no supervisado",
    "reinforcement learning", "refuerzo", "clasificación", "regresión", "clustering",
    "agrupamiento", "overfitting", "underfitting", "sesgo", "varianza", "bias", "variance",
    "precisión", "accuracy", "recall", "exhaustividad", "f1", "matriz de confusión",
    "curva roc", "auc", "cross-validation", "validación cruzada", "hiperparámetro",
    "feature", "característica", "preprocesamiento", "normalización", "estandarización",
    # Algoritmos y modelos
    "regresión lineal", "linear regression", "logistic", "logística", "svm", "máquina de vectores",
    "support vector", "árbol", "decision tree", "random forest", "gradient boosting", "xgboost",
    "k-means", "kmeans", "k nearest", "knn", "naive bayes", "perceptrón", "red neuronal",
    "neural network", "deep learning", "cnn", "rnn", "transformer", "autoencoder", "gan",
    # Scikit-Learn / Keras / TensorFlow
    "scikit-learn", "sklearn", "tensorflow", "keras", "tf", "pipeline", "estimator",
    "fit", "predict", "modelo", "entrenamiento", "epochs", "batch", "optimizer",
    "loss", "metric", "callback", "early stopping", "dropout", "regularización",
    "l1", "l2", "batch normalization", "data augmentation", "transfer learning",
    # Libro / capítulos
    "libro", "capítulo", "chapter", "página", "page", "aurélien", "geron",
]


def route(state: AgentState) -> dict:
    """Decide si es necesario recuperar contexto del libro."""
    question = state["question"].lower()
    needs_rag = any(keyword in question for keyword in RAG_KEYWORDS)
    return {"route": "RAG" if needs_rag else "DIRECTO"}


def retrieve(state: AgentState) -> dict:
    docs = retriever.invoke(state["question"])
    return {"context": "\n\n".join(doc.page_content for doc in docs)}


def generate(state: AgentState) -> dict:
    system_content = (
        DIRECT_PROMPT
        if state.get("route") == "DIRECTO"
        else SYSTEM_PROMPT.format(context=state["context"])
    )

    prompt_messages = [SystemMessage(content=system_content)]
    prompt_messages.extend(state.get("messages", []))
    prompt_messages.append(HumanMessage(content=state["question"]))

    response = llm.invoke(prompt_messages)
    return {
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=response.content),
        ]
    }


def build_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("route", route)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    workflow.set_entry_point("route")
    workflow.add_conditional_edges(
        "route",
        lambda state: state["route"],
        {"RAG": "retrieve", "DIRECTO": "generate"},
    )
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


agent = build_agent()
sessions: dict[str, List[BaseMessage]] = {}

app = FastAPI(title="ML Book Chat API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str


class ResetRequest(BaseModel):
    session_id: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = sessions.get(req.session_id, [])
    result = agent.invoke(
        {
            "messages": history,
            "context": "",
            "question": req.message,
            "route": "",
        }
    )
    sessions[req.session_id] = result["messages"]
    return ChatResponse(
        answer=result["messages"][-1].content,
        session_id=req.session_id,
    )


@app.post("/reset")
def reset(req: ResetRequest):
    sessions.pop(req.session_id, None)
    return {"status": "ok", "session_id": req.session_id}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
