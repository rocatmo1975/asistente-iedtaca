import streamlit as st
import os

# Importaciones de motor de IA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# IMPORTACIÓN CORREGIDA:
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. CONFIGURACIÓN DE PÁGINA E ICONO REFORZADO ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")
LOGO_URL_RAW = "https://github.com/rocatmo1975/asistente-iedtaca/blob/main/logo.png?raw=true"
NOMBRE_APP = "ASISTENTE IA IEDTACA"

st.set_page_config(
    page_title=NOMBRE_APP,
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🏫",
    layout="wide"
)

# INYECCIÓN DE CÓDIGO HTML PARA ICONO EN CELULARES
st.markdown(f"""
    <head>
        <link rel="icon" type="image/png" href="{LOGO_URL_RAW}">
        <link rel="apple-touch-icon" href="{LOGO_URL_RAW}">
        <meta name="mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-capable" content="yes">
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
else:
    st.markdown("<h3 style='text-align: center;'>🏫</h3>", unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: gray;'>Sistema de consulta técnica - Carmen de Ariguaní</p>", unsafe_allow_html=True)
st.markdown("---")

# --- 3. LÓGICA DE IA CON MEJORA DE RECUPERACIÓN (RAG) ---
api_key = st.secrets.get("OPENAI_API_KEY")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

@st.cache_resource(show_spinner="Analizando y optimizando la base de conocimiento... Esto mejorará las respuestas.")
def inicializar_ia(folder_path, _api_key):
    if not _api_key:
        return None
    
    os.environ["OPENAI_API_KEY"] = _api_key
    
    if not os.path.exists(folder_path):
        return None
    
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdf_files:
        return None
        
    try:
        documentos_completos = []
        for pdf in pdf_files:
            ruta_pdf = os.path.join(folder_path, pdf)
            loader = PyPDFLoader(ruta_pdf)
            documentos_completos.extend(loader.load())
        
        # DIVISIÓN DE TEXTO: Esto permite que la IA encuentre respuestas específicas
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        textos_fragmentados = text_splitter.split_documents(documentos_completos)
        
        vector_db = FAISS.from_documents(textos_fragmentados, OpenAIEmbeddings())
        
        # BUSCADOR MEJORADO: Recupera 10 fragmentos para mayor precisión
        retriever = vector_db.as_retriever(search_kwargs={"k": 10})
        
        model = ChatOpenAI(model="gpt-4o", temperature=0)
        
        template = """
        Eres el ASISTENTE IA IEDTACA, experto en la normativa de la institución.
        Tu misión es responder preguntas de docentes y directivos usando el contexto proporcionado.

        INSTRUCCIONES DE RESPUESTA:
        1. Si la información está en los documentos, explícala detalladamente.
        2. Si la información es parcial, intenta relacionar los datos para ayudar al docente.
        3. Si la respuesta NO está en los documentos, di: "Lamentablemente no encontré esa información específica en los manuales cargados, pero le sugiero consultar con [coordinación/secretaría] o revisar el documento de [tema relacionado]".
        
        Contexto: {context}
        Pregunta: {question}
        Respuesta:
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
        st.error(f"Error procesando documentos: {e}")
        return None

# --- 4. EJECUCIÓN DEL CHAT ---
if not api_key:
    st.error("❌ Falta la API KEY en los Secrets de Streamlit.")
else:
    rag_chain = inicializar_ia(DOCS_DIR, api_key)
    
    if rag_chain:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt_input := st.chat_input("¿Qué duda técnica o normativa tienes hoy?"):
            st.session_state.messages.append({"role": "user", "content": prompt_input})
            with st.chat_message("user"):
                st.markdown(prompt_input)

            with st.chat_message("assistant"):
                with st.spinner("Escaneando manuales institucionales..."):
                    try:
                        respuesta = rag_chain.invoke(prompt_input)
                        st.markdown(respuesta)
                        st.session_state.messages.append({"role": "assistant", "content": respuesta})
                    except Exception as e:
                        st.error(f"Error en la respuesta: {e}")
    else:
        st.warning("⚠️ No se encontraron documentos en la carpeta 'docs'.")
