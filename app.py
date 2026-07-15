import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import google.generativeai as genai

# 1. Configuración de API Key (Lee desde los Secrets de Streamlit)
# Streamlit maneja los secretos a través de st.secrets
api_key = st.secrets.get("LLM_API_KEY")
if not api_key:
    st.error("No se encontró LLM_API_KEY en los secretos de la aplicación.")
    st.stop()

genai.configure(api_key=api_key)

# Configuración de la página
st.set_page_config(page_title="RAG arXiv Papers", layout="wide")
st.title("Asistente RAG - arXiv Papers")
st.markdown(
    "Ingresa tu consulta sobre inteligencia artificial o ciencias de la computación. El sistema buscará en el corpus de arXiv."
)


# 2. Cargar datos y modelo en caché para que no se recargue en cada interacción
@st.cache_resource
def cargar_sistema():
    df1 = pd.read_csv("corpus/arxiv_data_210930-054931.csv")
    df2 = pd.read_csv("corpus/arxiv_data.csv").rename(
        columns={"summaries": "abstracts"}
    )
    column_order = ["titles", "abstracts", "terms"]
    corpus_df = pd.concat([df1[column_order], df2[column_order]], ignore_index=True)
    corpus_df = (
        corpus_df.dropna(subset=["titles", "abstracts"])
        .drop_duplicates(subset=["titles"])
        .reset_index(drop=True)
    )
    corpus_df["texto_completo"] = corpus_df["titles"] + ". " + corpus_df["abstracts"]

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings_file = "embedings/embeddings_corpus.npy"

    if os.path.exists(embeddings_file):
        embeddings_corpus = np.load(embeddings_file)
    else:
        embeddings_corpus = embedding_model.encode(corpus_df["texto_completo"].tolist())
        np.save(embeddings_file, embeddings_corpus)

    dimension = embeddings_corpus.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.float32(embeddings_corpus))

    return corpus_df, embedding_model, index


corpus_df, embedding_model, index = cargar_sistema()


# 3. Funciones RAG
def recuperar_contexto(query, k=3, umbral_distancia=1.2):
    query_embedding = np.float32(embedding_model.encode([query]))
    distancias, indices = index.search(query_embedding, k)

    contexto_texto = ""
    evidencias = []

    for i, idx in enumerate(indices[0]):
        score = float(distancias[0][i])
        if score <= umbral_distancia:
            doc = corpus_df.iloc[idx]
            evidencias.append(
                {"score": score, "title": doc["titles"], "abstract": doc["abstracts"]}
            )
            contexto_texto += (
                f"Título: {doc['titles']}\nAbstract: {doc['abstracts']}\n---\n"
            )

    return (contexto_texto, evidencias) if evidencias else (None, [])


def generar_respuesta_rag(query, contexto):
    if not contexto:
        return "Lo siento, el corpus actual no contiene información suficiente para responder a esta consulta."

    system_prompt = "Eres un asistente académico experto. Responde utilizando ÚNICAMENTE la información en el contexto extraído. Si no hay información suficiente, indícalo claramente."
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_prompt,
        generation_config=genai.GenerationConfig(temperature=0.2),
    )
    user_message = f"Contexto:\n{contexto}\n\nPregunta: {query}"

    try:
        return model.generate_content(user_message).text
    except Exception as e:
        return f"Error de API: {str(e)}"


# 4. Interfaz de Usuario
query = st.text_input(
    "Consulta:",
    placeholder="Ej. What are the main applications of Graph Neural Networks?",
)

if st.button("Enviar"):
    if query:
        with st.spinner("Buscando en el corpus y generando respuesta..."):
            contexto, evidencias = recuperar_contexto(query)
            respuesta = generar_respuesta_rag(query, contexto)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Respuesta")
                st.write(respuesta)

            with col2:
                st.subheader("Fuentes Utilizadas")
                if not evidencias:
                    st.warning("No se encontraron documentos relevantes.")
                else:
                    for i, doc in enumerate(evidencias):
                        st.markdown(f"**{i+1}. {doc['title']}**")
                        st.caption(f"Similitud L2: {doc['score']:.4f}")
                        st.write(f"> {doc['abstract'][:250]}...")
                        st.divider()
    else:
        st.warning("Por favor, ingresa una consulta.")
