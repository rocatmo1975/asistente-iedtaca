import streamlit as st
import os

# Importaciones de motor de IA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. CONFIGURACIÓN DE PÁGINA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")
LOGO_URL_RAW = "https://github.com/rocatmo1975/asistente-iedtaca/blob/main/logo.png?raw=true"
NOMBRE_APP = "ASISTENTE IA IEDTACA"

st.set_page_config(
    page_title=NOMBRE_APP,
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🏫",
    layout="wide"
)

# ESTILOS E ICONOS
st.markdown(f"""
    <head>
        <link rel="icon" type="image/png" href="{LOGO_URL_RAW}">
        <link rel="apple-touch-icon" href="{LOGO_URL_RAW}">
    </head>
    <style>
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .stChatMessage {{ border-radius: 15px; }}
    </style>
""", unsafe_allow_html=True)

# --- 2. DISEÑO DE INTERFAZ ---
st.markdown(f"<h1 style='text-align: center;'>{NOMBRE_APP}</h1>", unsafe_allow_html=True)

if os.path.exists(LOGO_PATH):
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        try:
            st.image(LOGO_PATH, use_container_width=True)
        except:
            st.write("🏫")
st.markdown("<p style='text-align: center; color: gray;'>Versión Profesional de Alta Precisión</p>", unsafe_allow_html=True)
st.markdown("---")

# --- 3. LÓGICA DE IA CON BUSCADOR DE ALTA PRECISIÓN (MMR) ---
api_key = st.secrets.get("OPENAI_API_KEY")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

@st.cache_resource(show_spinner="Optimizando búsqueda en 61 documentos... Por favor espere.")
def inicializar_ia(folder_path, _api_key):
    if not _api_key: return None
    os.environ["OPENAI_API_KEY"] = _api_key
    
    if not os.path.exists(folder_path): return None
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdf_files: return None
        
    try:
        documentos_completos = []
        for pdf in pdf_files:
            loader = PyPDFLoader(os.path.join(folder_path, pdf))
            documentos_completos.extend(loader.load())
        
        # Fragmentación más fina para no saltarse nombres propios
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        textos_fragmentados = text_splitter.split_documents(documentos_completos)
        vector_db = FAISS.from_documents(textos_fragmentados, OpenAIEmbeddings())
        
        # --- AJUSTE MAESTRO: BUSCADOR MMR ---
        # fetch_k=20 analiza 20 bloques y elige los 7 más relevantes y distintos
        retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 7, "fetch_k": 20, "lambda_mult": 0.5}
        )
        
        model = ChatOpenAI(model="gpt-4o", temperature=0)
        
        template = """
        Eres el Asistente Experto de la IEDTACA. Tu misión es extraer datos exactos de los manuales.

        CONTEXTO INSTITUCIONAL:
        {context}

        PREGUNTA DEL DOCENTE:
        {question}

        INSTRUCCIONES:
        1. Si la pregunta pide un nombre propio (ej: Rector, Coordinador), búscalo con prioridad en los encabezados o firmas detectadas en el texto.
        2. Si encuentras la información, responde de forma directa y profesional.
        3. Si la información no es exacta pero hay algo muy relacionado, menciónalo.
        
        RESPUESTA:
        """
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | model | StrOutputParser()
        )
        return chain
    except Exception as e:
        st.error(f"Error técnico: {e}")
        return None

# --- 4. EJECUCIÓN ---
if not api_key:
    st.error("Falta la API KEY.")
else:
    rag_chain = inicializar_ia(DOCS_DIR, api_key)
    
    if rag_chain:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt_input := st.chat_input("¿Qué dato específico buscas hoy?"):
            st.session_state.messages.append({"role": "user", "content": prompt_input})
            with st.chat_message("user"):
                st.markdown(prompt_input)

            with st.chat_message("assistant"):
                with st.spinner("Escaneando minuciosamente..."):
                    respuesta = rag_chain.invoke(prompt_input)
                    st.markdown(respuesta)
                    st.session_state.messages.append({"role": "assistant", "content": respuesta})
