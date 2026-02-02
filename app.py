import streamlit as st
import os

# Importaciones de motor de IA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. CONFIGURACIÓN ---
NOMBRE_APP = "ASISTENTE IA IEDTACA"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Intentamos buscar el logo, pero no dejaremos que la app falle por esto
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")

st.set_page_config(page_title=NOMBRE_APP, page_icon="🏫", layout="wide")

# --- 2. DISEÑO DE INTERFAZ (MODO SEGURO) ---
st.markdown(f"<h1 style='text-align: center;'>{NOMBRE_APP}</h1>", unsafe_allow_html=True)

# Intentar mostrar logo sin romper la app
if os.path.exists(LOGO_PATH):
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st.image(LOGO_PATH, use_container_width=True)

st.markdown("<p style='text-align: center; color: gray;'>Sistema de consulta técnica - Carmen de Ariguaní</p>", unsafe_allow_html=True)
st.markdown("---")

# --- 3. LÓGICA AUTOMÁTICA ---
api_key = st.secrets.get("OPENAI_API_KEY")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

if not api_key:
    st.error("❌ Falta la OPENAI_API_KEY en los Secrets de Streamlit.")
else:
    os.environ["OPENAI_API_KEY"] = api_key
    
    if os.path.exists(DOCS_DIR):
        pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
        
        if not pdf_files:
            st.info("Esperando archivos PDF en la carpeta 'docs'...")
        else:
            with st.spinner("Cargando base de conocimiento..."):
                try:
                    paginas = []
                    for pdf in pdf_files:
                        loader = PyPDFLoader(os.path.join(DOCS_DIR, pdf))
                        paginas.extend(loader.load())
                    
                    vector_db = FAISS.from_documents(paginas, OpenAIEmbeddings())
                    retriever = vector_db.as_retriever()
                    
                    model = ChatOpenAI(model="gpt-4o", temperature=0)
                    
                    template = """
                    Eres el ASISTENTE IA IEDTACA. Responde basándote en el contexto institucional.
                    Contexto: {context}
                    Pregunta: {question}
                    """
                    prompt = ChatPromptTemplate.from_template(template)

                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)

                    rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt | model | StrOutputParser()
                    )

                    st.success(f"✅ {len(pdf_files)} documentos institucionales listos.")
                    
                    # --- 4. CHAT ---
                    if "messages" not in st.session_state:
                        st.session_state.messages = []

                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                    if prompt_input := st.chat_input("¿En qué puedo ayudarte hoy?"):
                        st.session_state.messages.append({"role": "user", "content": prompt_input})
                        with st.chat_message("user"):
                            st.markdown(prompt_input)

                        with st.chat_message("assistant"):
                            respuesta = rag_chain.invoke(prompt_input)
                            st.markdown(respuesta)
                        st.session_state.messages.append({"role": "assistant", "content": respuesta})
                
                except Exception as e:
                    st.error(f"Error en el procesamiento: {e}")
    else:
        st.error(f"No se encuentra la carpeta 'docs'. Créala en GitHub y sube tus archivos allí.")
