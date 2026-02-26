"""
=============================================================
  RAG RECENSEMENT — Serveur Web (FastAPI)
=============================================================
  Backend de l'application de chat RAG.
  Expose une API REST consommée par l'interface web.

  VARIABLES D'ENVIRONNEMENT À DÉFINIR :
    ALBERT_API_KEY   : votre clé API Albert
    ALBERT_BASE_URL  : https://albert.api.etalab.gouv.fr/v1
    ALBERT_MODEL     : mistralai/Mistral-Small-3.2-24B-Instruct-2506
    DOCS_DIR         : chemin vers le dossier de documents
    FAISS_INDEX      : nom du dossier de la base vectorielle

  LANCEMENT LOCAL :
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
=============================================================
"""

import os, zipfile, re, threading, asyncio
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lxml import etree
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from ddgs import DDGS

# ─────────────────────────────────────────────────────────────
# CONFIGURATION (via variables d'environnement ou valeurs par défaut)
# ─────────────────────────────────────────────────────────────

ALBERT_API_KEY  = os.getenv("ALBERT_API_KEY",  "COLLEZ_VOTRE_CLE_ALBERT_ICI")
ALBERT_BASE_URL = os.getenv("ALBERT_BASE_URL", "https://albert.api.etalab.gouv.fr/v1")
ALBERT_MODEL    = os.getenv("ALBERT_MODEL",    "mistralai/Mistral-Small-3.2-24B-Instruct-2506")
DOCS_DIR        = os.getenv("DOCS_DIR",        "./documents")
FAISS_INDEX     = os.getenv("FAISS_INDEX",     "faiss_index_recensement")

# ─────────────────────────────────────────────────────────────
# FONCTIONS D'EXTRACTION DE DOCUMENTS
# ─────────────────────────────────────────────────────────────

NS_DRAW = 'urn:oasis:names:tc:opendocument:xmlns:drawing:1.0'

def extraire_tout_texte(element):
    return ' '.join(p.strip() for p in element.itertext() if p.strip())

def charger_odp_lxml(chemin):
    docs = []
    try:
        with zipfile.ZipFile(chemin, 'r') as z:
            if 'content.xml' not in z.namelist():
                return []
            with z.open('content.xml') as f:
                tree = etree.parse(f)
        pages = tree.findall(f'.//{{{NS_DRAW}}}page')
        for i, page in enumerate(pages, start=1):
            texte = extraire_tout_texte(page).strip()
            if texte:
                docs.append(Document(
                    page_content=texte,
                    metadata={"source": os.path.basename(chemin), "slide": i, "type": "odp"}
                ))
    except Exception as e:
        print(f"  [WARN] ODP non lisible ({os.path.basename(chemin)}) : {e}")
    return docs

def charger_dossier(dossier):
    tous = []
    if not os.path.isdir(dossier):
        print(f"  [WARN] Dossier introuvable : {dossier}")
        return tous
    for nom in sorted(os.listdir(dossier)):
        chemin = os.path.join(dossier, nom)
        ext = nom.lower().rsplit(".", 1)[-1] if "." in nom else ""
        try:
            if ext == "pdf":
                docs = PyPDFLoader(chemin).load()
                for d in docs:
                    d.metadata["type"] = "pdf"
                print(f"  [OK] PDF : {nom} ({len(docs)} pages)")
            elif ext == "odp":
                docs = charger_odp_lxml(chemin)
                print(f"  [OK] ODP : {nom} ({len(docs)} slides)")
            else:
                continue
            tous.extend(docs)
        except Exception as e:
            print(f"  [SKIP] {nom} : {e}")
    return tous

# ─────────────────────────────────────────────────────────────
# CLASSE RETRIEVER HYBRIDE (BM25 + Vectoriel)
# ─────────────────────────────────────────────────────────────

class RetrieverHybride:
    def __init__(self, vectorstore, bm25, k=10):
        self.vs  = vectorstore
        self.bm25 = bm25
        self.k    = k

    def invoke(self, question):
        res_vec  = self.vs.similarity_search(question, k=self.k)
        res_bm25 = self.bm25.invoke(question)[:self.k]
        vus, uniques = set(), []
        for d in res_vec + res_bm25:
            cle = d.page_content[:80]
            if cle not in vus:
                vus.add(cle)
                uniques.append(d)
        return uniques[:self.k]

# ─────────────────────────────────────────────────────────────
# ÉTAT GLOBAL DE L'APPLICATION
# ─────────────────────────────────────────────────────────────

class AppState:
    llm = None
    retriever = None
    chaine_rag = None
    chaine_web = None
    chaine_llm = None
    chaine_routeur = None
    pret = False
    message_init = ""

state = AppState()

# ─────────────────────────────────────────────────────────────
# INITIALISATION AU DÉMARRAGE
# ─────────────────────────────────────────────────────────────

def initialiser_rag_background():
    """Lance l'initialisation du RAG dans un thread séparé pour ne pas bloquer le démarrage du serveur."""
    print("\n" + "="*60)
    print("  RAG RECENSEMENT — Initialisation en arrière-plan...")
    print("="*60)
    try:
        # LLM Albert API
        state.llm = ChatOpenAI(
            model=ALBERT_MODEL,
            temperature=0,
            api_key=ALBERT_API_KEY,
            base_url=ALBERT_BASE_URL
        )

        # Embeddings locaux
        print("[INFO] Chargement du modèle d'embedding...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # Base vectorielle
        if os.path.exists(FAISS_INDEX):
            print(f"[INFO] Chargement de la base vectorielle existante : {FAISS_INDEX}")
            vectorstore = FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)
            # Reconstruire BM25 depuis les documents du vectorstore
            docs_pour_bm25 = list(vectorstore.docstore._dict.values())
            bm25 = BM25Retriever.from_documents(docs_pour_bm25, k=10)
        else:
            print(f"[INFO] Construction de la base vectorielle depuis : {DOCS_DIR}")
            documents = charger_dossier(DOCS_DIR)
            if not documents:
                state.message_init = "Aucun document trouvé dans le dossier configuré."
                state.pret = False
                yield
                return
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            chunks = splitter.split_documents(documents)
            print(f"[INFO] {len(chunks)} chunks créés, vectorisation en cours...")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local(FAISS_INDEX)
            bm25 = BM25Retriever.from_documents(chunks, k=10)
            print(f"[INFO] Base sauvegardée : {FAISS_INDEX}")

        # Retriever hybride
        retriever = RetrieverHybride(vectorstore, bm25, k=10)

        def recuperer_et_formater(question):
            docs = retriever.invoke(question)
            return "\n---\n".join(
                f"[{d.metadata.get('source','?')}, "
                f"{'slide' if d.metadata.get('type')=='odp' else 'page'} "
                f"{d.metadata.get('slide', d.metadata.get('page','?'))}]\n{d.page_content}"
                for d in docs
            )

        # Chaîne RAG
        prompt_rag = ChatPromptTemplate.from_template("""Tu es un assistant expert du recensement de la population française.
Réponds en français de manière précise et structurée, en te basant sur les extraits de documents fournis.
Si les extraits contiennent des éléments pertinents, synthétise-les pour répondre à la question.
Cite les sources (nom du fichier) quand c'est utile.

Extraits de documents :
{context}

Question : {question}
Réponse :""")

        state.chaine_rag = (
            {"context": RunnableLambda(recuperer_et_formater), "question": RunnablePassthrough()}
            | prompt_rag
            | state.llm
            | StrOutputParser()
        )

        # Chaîne Web
        def recherche_web(question):
            try:
                with DDGS() as ddgs:
                    resultats = list(ddgs.text(question + " site:legifrance.gouv.fr OR site:insee.fr OR site:service-public.fr", max_results=5))
                if not resultats:
                    resultats = list(DDGS().text(question, max_results=5))
                return "\n---\n".join(
                    f"[{r.get('title','')}]\n{r.get('body','')}\nSource : {r.get('href','')}"
                    for r in resultats
                )
            except Exception as e:
                return f"Recherche web indisponible : {e}"

        prompt_web = ChatPromptTemplate.from_template("""Tu es un assistant expert du recensement de la population française.
Réponds en français en te basant sur les résultats de recherche web fournis.
Cite les sources (URLs) quand c'est pertinent.

Résultats web :
{context}

Question : {question}
Réponse :""")

        state.chaine_web = (
            {"context": RunnableLambda(recherche_web), "question": RunnablePassthrough()}
            | prompt_web
            | state.llm
            | StrOutputParser()
        )

        # Chaîne LLM général
        prompt_llm = ChatPromptTemplate.from_template("""Tu es un assistant expert du recensement de la population française et de la rédaction administrative.
Réponds en français de manière professionnelle et structurée.
Pour les courriers, utilise les formules de politesse appropriées au contexte administratif français.

Question : {question}
Réponse :""")

        state.chaine_llm = (
            {"question": RunnablePassthrough()}
            | prompt_llm
            | state.llm
            | StrOutputParser()
        )

        # Routeur
        prompt_routeur = ChatPromptTemplate.from_template("""Analyse cette question et réponds UNIQUEMENT par un seul mot parmi : RAG, WEB, LLM

- RAG : question sur des procédures internes, formations, consignes, tournées, questionnaires, rôles des agents, coordonnateurs
- WEB : question sur des décrets, lois, dates officielles, actualités, textes réglementaires disponibles sur internet
- LLM : rédaction de courriers, reformulations, traductions, calculs, questions générales sans lien avec les documents

Question : {question}
Réponse (un seul mot) :""")

        state.chaine_routeur = (
            {"question": RunnablePassthrough()}
            | prompt_routeur
            | state.llm
            | StrOutputParser()
        )

        state.pret = True
        print("\n[OK] RAG prêt.\n")
    except Exception as e:
        state.message_init = f"Erreur d'initialisation : {e}"
        print(f"[ERREUR] {e}")
        state.pret = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lancer l'initialisation dans un thread séparé pour ne pas bloquer le port
    thread = threading.Thread(target=initialiser_rag_background, daemon=True)
    thread.start()
    yield  # Le serveur est immédiatement disponible, le RAG se charge en arrière-plani

# ─────────────────────────────────────────────────────────────
# APPLICATION FASTAPI
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Recensement", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ─────────────────────────────────────────────────────────────
# MODÈLES DE DONNÉES
# ─────────────────────────────────────────────────────────────

class MessageRequest(BaseModel):
    question: str
    mode_force: str = "AUTO"  # AUTO, RAG, WEB, LLM

class MessageResponse(BaseModel):
    reponse: str
    mode: str
    sources: List[str] = []

# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("templates/index.html")

@app.get("/health")
async def health():
    return {"status": "ok" if state.pret else "initialisation", "message": state.message_init}

@app.post("/chat", response_model=MessageResponse)
async def chat(req: MessageRequest):
    if not state.pret:
        raise HTTPException(status_code=503, detail="Le RAG est en cours d'initialisation, veuillez patienter.")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    # Détermination du mode
    mode = req.mode_force.upper()
    if mode == "AUTO":
        try:
            mode_brut = state.chaine_routeur.invoke(question).strip().upper()
            mode = "RAG" if "RAG" in mode_brut else "WEB" if "WEB" in mode_brut else "LLM"
        except Exception:
            mode = "RAG"

    # Génération de la réponse
    try:
        if mode == "RAG":
            reponse = state.chaine_rag.invoke(question)
        elif mode == "WEB":
            reponse = state.chaine_web.invoke(question)
        else:
            reponse = state.chaine_llm.invoke(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération : {str(e)}")

    return MessageResponse(reponse=reponse, mode=mode)
