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
    EMBED_MODEL      : BAAI/bge-m3
    DOCS_DIR         : chemin vers le dossier de documents (défaut: ./documents)
    FAISS_INDEX      : nom du dossier de la base vectorielle (défaut: faiss_index_recensement)
    UPLOAD_SECRET    : mot de passe pour protéger la page d'upload (optionnel)

  LANCEMENT LOCAL :
    python app.py
=============================================================
"""

import os, zipfile, re, threading, shutil
from typing import List
from contextlib import asynccontextmanager

import requests as _requests
from langchain_core.embeddings import Embeddings as _LCEmbeddings

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lxml import etree
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


# ─────────────────────────────────────────────────────────────
# CLASSE D'EMBEDDINGS PERSONNALISÉE POUR ALBERT API
# (contourne le SDK openai qui force encoding_format=base64)
# ─────────────────────────────────────────────────────────────

class AlbertEmbeddings(_LCEmbeddings):
    """
    Classe d'embeddings qui appelle directement l'API Albert
    avec encoding_format='float', sans passer par le SDK openai
    qui force base64 par défaut.
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        self._api_key  = api_key
        self._base_url = base_url.rstrip("/")
        self._model    = model
        self._headers  = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Appel HTTP direct à /embeddings avec encoding_format=float."""
        # Traitement par lots de 32 pour éviter les timeouts
        all_embeddings = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {
                "model": self._model,
                "input": batch,
                "encoding_format": "float",
            }
            resp = _requests.post(
                f"{self._base_url}/embeddings",
                headers=self._headers,
                json=payload,
                timeout=120,
            )
            if not resp.ok:
                raise RuntimeError(
                    f"Albert Embeddings API error {resp.status_code}: {resp.text[:500]}"
                )
            data = resp.json()
            # Trier par index pour garantir l'ordre
            items = sorted(data["data"], key=lambda x: x["index"])
            all_embeddings.extend(item["embedding"] for item in items)
        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

# ─────────────────────────────────────────────────────────────
# CONFIGURATION (via variables d'environnement ou valeurs par défaut)
# ─────────────────────────────────────────────────────────────

ALBERT_API_KEY  = os.getenv("ALBERT_API_KEY",  "COLLEZ_VOTRE_CLE_ALBERT_ICI")
ALBERT_BASE_URL = os.getenv("ALBERT_BASE_URL", "https://albert.api.etalab.gouv.fr/v1")
ALBERT_MODEL    = os.getenv("ALBERT_MODEL",    "mistralai/Mistral-Small-3.2-24B-Instruct-2506")
EMBED_MODEL     = os.getenv("EMBED_MODEL",     "BAAI/bge-m3")
DOCS_DIR        = os.getenv("DOCS_DIR",        "./documents")
FAISS_INDEX     = os.getenv("FAISS_INDEX",     "faiss_index_recensement")
UPLOAD_SECRET   = os.getenv("UPLOAD_SECRET",   "")   # laisser vide = pas de protection

# Extensions de fichiers supportées pour l'indexation
EXTENSIONS_SUPPORTEES = ('.pdf', '.odp', '.odt', '.xls', '.xlsx', '.ods')

# Création automatique des dossiers nécessaires au démarrage
os.makedirs("static",    exist_ok=True)
os.makedirs("documents", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# FONCTIONS D'EXTRACTION DE DOCUMENTS
# ─────────────────────────────────────────────────────────────

NS_DRAW = 'urn:oasis:names:tc:opendocument:xmlns:drawing:1.0'

def extraire_tout_texte(element):
    return ' '.join(p.strip() for p in element.itertext() if p.strip())

def charger_odp_lxml(chemin):
    """Extrait le texte d'un fichier ODP (présentation LibreOffice Impress)."""
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


def charger_odt(chemin):
    """Extrait le texte d'un fichier ODT (document texte LibreOffice Writer)."""
    docs = []
    try:
        from odf.opendocument import load as odf_load
        from odf import teletype

        doc_odf = odf_load(chemin)
        contenu = teletype.extractText(doc_odf.text)
        if contenu.strip():
            docs.append(Document(
                page_content=contenu.strip(),
                metadata={"source": os.path.basename(chemin), "page": 1, "type": "odt"}
            ))
    except Exception as e:
        print(f"  [WARN] ODT non lisible ({os.path.basename(chemin)}) : {e}")
    return docs


def charger_tableur(chemin):
    """
    Extrait le texte d'un fichier tableur XLS ou XLSX (Excel).
    Chaque feuille devient un Document distinct.
    Note : les fichiers .ods sont traités par charger_ods() via odfpy.
    """
    docs = []
    nom = os.path.basename(chemin)
    ext = nom.lower().rsplit(".", 1)[-1]
    try:
        if ext == "xls":
            # Format binaire Microsoft ancien (.xls) → xlrd
            import xlrd
            wb = xlrd.open_workbook(chemin)
            for sheet in wb.sheets():
                lignes = []
                for row_idx in range(sheet.nrows):
                    cellules = [str(sheet.cell_value(row_idx, col)).strip()
                                for col in range(sheet.ncols)]
                    ligne = "\t".join(c for c in cellules if c)
                    if ligne.strip():
                        lignes.append(ligne)
                contenu = "\n".join(lignes)
                if contenu.strip():
                    docs.append(Document(
                        page_content=contenu,
                        metadata={"source": nom, "sheet": sheet.name, "type": "xls"}
                    ))
        elif ext == "xlsx":
            # Format XML Microsoft (.xlsx) → openpyxl
            import openpyxl
            wb = openpyxl.load_workbook(chemin, read_only=True, data_only=True)
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                lignes = []
                for row in ws.iter_rows(values_only=True):
                    cellules = [str(c).strip() for c in row if c is not None and str(c).strip()]
                    if cellules:
                        lignes.append("\t".join(cellules))
                contenu = "\n".join(lignes)
                if contenu.strip():
                    docs.append(Document(
                        page_content=contenu,
                        metadata={"source": nom, "sheet": sheet_name, "type": "xlsx"}
                    ))
            wb.close()
    except Exception as e:
        print(f"  [WARN] Tableur non lisible ({nom}) : {e}")
    return docs


def charger_ods(chemin):
    """
    Extrait le texte d'un fichier ODS (LibreOffice Calc) via odfpy.
    openpyxl ne supporte pas ce format malgré ce que sa doc indique.
    Chaque feuille devient un Document distinct.
    """
    docs = []
    nom = os.path.basename(chemin)
    try:
        from odf.opendocument import load as odf_load
        from odf import teletype
        from odf.table import Table, TableRow, TableCell

        doc_odf = odf_load(chemin)
        sheets = doc_odf.spreadsheet.getElementsByType(Table)
        for sheet in sheets:
            sheet_name = sheet.getAttribute("name") or "Feuille"
            rows = sheet.getElementsByType(TableRow)
            lignes = []
            for row in rows:
                cells = row.getElementsByType(TableCell)
                cellules = []
                for cell in cells:
                    texte = teletype.extractText(cell).strip()
                    if texte:
                        cellules.append(texte)
                if cellules:
                    lignes.append("\t".join(cellules))
            contenu = "\n".join(lignes)
            if contenu.strip():
                docs.append(Document(
                    page_content=contenu,
                    metadata={"source": nom, "sheet": sheet_name, "type": "ods"}
                ))
    except Exception as e:
        print(f"  [WARN] ODS non lisible ({nom}) : {e}")
    return docs


def charger_dossier(dossier):
    """
    Charge récursivement tous les documents supportés depuis `dossier`
    et ses sous-dossiers (PDF, ODP, ODT, XLS, XLSX, ODS).
    """
    tous = []
    if not os.path.isdir(dossier):
        print(f"  [WARN] Dossier introuvable : {dossier}")
        return tous

    # os.walk() parcourt le dossier racine ET tous ses sous-dossiers
    fichiers_trouves = []
    for racine, sous_dossiers, fichiers in os.walk(dossier):
        sous_dossiers.sort()   # ordre alphabétique pour la reproductibilité
        for nom in sorted(fichiers):
            if nom.lower().endswith(EXTENSIONS_SUPPORTEES):
                fichiers_trouves.append(os.path.join(racine, nom))

    if not fichiers_trouves:
        print(f"  [WARN] Aucun fichier {EXTENSIONS_SUPPORTEES} dans : {dossier} (y compris sous-dossiers)")
        return tous

    print(f"  [INFO] {len(fichiers_trouves)} fichier(s) trouvé(s) dans {dossier} (récursif)")

    for chemin in fichiers_trouves:
        nom = os.path.basename(chemin)
        # Chemin relatif pour les métadonnées (plus lisible)
        chemin_relatif = os.path.relpath(chemin, dossier)
        ext = nom.lower().rsplit(".", 1)[-1]
        try:
            if ext == "pdf":
                docs = PyPDFLoader(chemin).load()
                for d in docs:
                    d.metadata["type"]   = "pdf"
                    d.metadata["source"] = chemin_relatif
                print(f"  [OK] PDF : {chemin_relatif} ({len(docs)} pages)")

            elif ext == "odp":
                docs = charger_odp_lxml(chemin)
                for d in docs:
                    d.metadata["source"] = chemin_relatif
                print(f"  [OK] ODP : {chemin_relatif} ({len(docs)} slides)")

            elif ext == "odt":
                docs = charger_odt(chemin)
                for d in docs:
                    d.metadata["source"] = chemin_relatif
                print(f"  [OK] ODT : {chemin_relatif} ({len(docs)} doc(s))")

            elif ext in ("xls", "xlsx"):
                docs = charger_tableur(chemin)
                for d in docs:
                    d.metadata["source"] = chemin_relatif
                print(f"  [OK] {ext.upper()} : {chemin_relatif} ({len(docs)} feuille(s))")

            elif ext == "ods":
                docs = charger_ods(chemin)
                for d in docs:
                    d.metadata["source"] = chemin_relatif
                print(f"  [OK] ODS : {chemin_relatif} ({len(docs)} feuille(s))")

            else:
                continue

            tous.extend(docs)
        except Exception as e:
            print(f"  [SKIP] {chemin_relatif} : {e}")

    return tous

# ─────────────────────────────────────────────────────────────
# CLASSE RETRIEVER HYBRIDE (BM25 + Vectoriel)
# ─────────────────────────────────────────────────────────────

class RetrieverHybride:
    def __init__(self, vectorstore, bm25, k=10):
        self.vs   = vectorstore
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
    llm             = None
    retriever       = None
    chaine_rag      = None
    chaine_web      = None
    chaine_llm      = None
    chaine_routeur  = None
    pret            = False
    en_cours        = False   # True pendant la (re)construction du RAG
    message_init    = ""
    nb_docs         = 0
    nb_chunks       = 0

state = AppState()

# ─────────────────────────────────────────────────────────────
# INITIALISATION DU RAG (peut être appelée plusieurs fois)
# ─────────────────────────────────────────────────────────────

def _construire_rag(forcer_reconstruction: bool = False):
    """
    Construit ou recharge le RAG.
    forcer_reconstruction=True : ignore le FAISS existant et repart des documents.
    """
    print("\n" + "="*60)
    print("  RAG RECENSEMENT — Initialisation...")
    print("="*60)

    state.pret       = False
    state.en_cours   = True
    state.message_init = "Construction en cours..."

    try:
        # LLM Albert API
        state.llm = ChatOpenAI(
            model=ALBERT_MODEL,
            temperature=0,
            api_key=ALBERT_API_KEY,
            base_url=ALBERT_BASE_URL
        )

        print("[INFO] Initialisation des embeddings (AlbertEmbeddings avec encoding_format=float)...")
        embeddings = AlbertEmbeddings(
            api_key=ALBERT_API_KEY,
            base_url=ALBERT_BASE_URL,
            model=EMBED_MODEL,
        )

        # ── Recoller les morceaux de index.faiss si nécessaire ──
        faiss_file   = os.path.join(FAISS_INDEX, "index.faiss")
        manifest_file = faiss_file + ".manifest"
        if os.path.exists(manifest_file) and not os.path.exists(faiss_file):
            print("[INFO] Reconstitution de index.faiss depuis les morceaux...")
            with open(manifest_file) as mf:
                nb_parts = int(mf.read().strip())
            with open(faiss_file, "wb") as out:
                for i in range(nb_parts):
                    part_path = f"{faiss_file}.part{i:02d}"
                    with open(part_path, "rb") as pf:
                        out.write(pf.read())
            print(f"[INFO] index.faiss reconstitué ({nb_parts} morceaux)")

        # Base vectorielle
        if not forcer_reconstruction and os.path.exists(FAISS_INDEX) and os.path.exists(faiss_file):
            print(f"[INFO] Chargement de la base vectorielle existante : {FAISS_INDEX}")
            vectorstore = FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)
            # Reconstituer le BM25 depuis les documents du vectorstore
            docs_pour_bm25 = list(vectorstore.docstore._dict.values())
            bm25 = BM25Retriever.from_documents(docs_pour_bm25, k=10)
            state.nb_docs   = len(set(d.metadata.get("source", "") for d in docs_pour_bm25))
            state.nb_chunks = len(docs_pour_bm25)
        else:
            print(f"[INFO] Construction depuis : {DOCS_DIR}")
            documents = charger_dossier(DOCS_DIR)
            if not documents:
                state.message_init = (
                    "Aucun document trouvé. "
                    "Uploadez vos fichiers PDF/ODP/ODT/XLS/XLSX/ODS via la page /admin."
                )
                state.pret     = False
                state.en_cours = False
                return

            state.nb_docs = len(set(d.metadata.get("source", "") for d in documents))
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            chunks = splitter.split_documents(documents)
            state.nb_chunks = len(chunks)
            print(f"[INFO] {len(chunks)} chunks créés, vectorisation en cours...")

            # Supprimer l'ancien index si présent
            if os.path.exists(FAISS_INDEX):
                shutil.rmtree(FAISS_INDEX)

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
                f"{'slide' if d.metadata.get('type')=='odp' else 'feuille' if d.metadata.get('type') in ('xls','xlsx','ods') else 'page'} "
                f"{d.metadata.get('slide', d.metadata.get('sheet', d.metadata.get('page','?')))}]\n{d.page_content}"
                for d in docs
            )

        # Chaîne RAG
        prompt_rag = ChatPromptTemplate.from_template(
            """Tu es un assistant expert du recensement de la population française.
Réponds en français de manière précise et structurée, en te basant sur les extraits de documents fournis.
Si les extraits contiennent des éléments pertinents, synthétise-les pour répondre à la question.
Cite les sources (nom du fichier) quand c'est utile.

Extraits de documents :
{context}

Question : {question}
Réponse :"""
        )
        state.chaine_rag = (
            {"context": RunnableLambda(recuperer_et_formater), "question": RunnablePassthrough()}
            | prompt_rag | state.llm | StrOutputParser()
        )

        # Chaîne Web (DuckDuckGo)
        def recherche_web(question):
            try:
                from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    resultats = list(ddgs.text(
                        question + " site:legifrance.gouv.fr OR site:insee.fr OR site:service-public.fr",
                        max_results=5
                    ))
                if not resultats:
                    with DDGS() as ddgs2:
                        resultats = list(ddgs2.text(question, max_results=5))
                return "\n---\n".join(
                    f"[{r.get('title','')}]\n{r.get('body','')}\nSource : {r.get('href','')}"
                    for r in resultats
                )
            except Exception as e:
                return f"Recherche web indisponible : {e}"

        prompt_web = ChatPromptTemplate.from_template(
            """Tu es un assistant expert du recensement de la population française.
Réponds en français en te basant sur les résultats de recherche web fournis.
Cite les sources (URLs) quand c'est pertinent.

Résultats web :
{context}

Question : {question}
Réponse :"""
        )
        state.chaine_web = (
            {"context": RunnableLambda(recherche_web), "question": RunnablePassthrough()}
            | prompt_web | state.llm | StrOutputParser()
        )

        # Chaîne LLM général
        prompt_llm = ChatPromptTemplate.from_template(
            """Tu es un assistant expert du recensement de la population française et de la rédaction administrative.
Réponds en français de manière professionnelle et structurée.
Pour les courriers, utilise les formules de politesse appropriées au contexte administratif français.

Question : {question}
Réponse :"""
        )
        state.chaine_llm = (
            {"question": RunnablePassthrough()}
            | prompt_llm | state.llm | StrOutputParser()
        )

        # Routeur
        prompt_routeur = ChatPromptTemplate.from_template(
            """Analyse cette question et réponds UNIQUEMENT par un seul mot parmi : RAG, WEB, LLM

- RAG : question sur des procédures internes, formations, consignes, tournées, questionnaires, rôles des agents, coordonnateurs
- WEB : question sur des décrets, lois, dates officielles, actualités, textes réglementaires disponibles sur internet
- LLM : rédaction de courriers, reformulations, traductions, calculs, questions générales sans lien avec les documents

Question : {question}
Réponse (un seul mot) :"""
        )
        state.chaine_routeur = (
            {"question": RunnablePassthrough()}
            | prompt_routeur | state.llm | StrOutputParser()
        )

        state.pret         = True
        state.message_init = ""
        print("\n[OK] RAG prêt.\n")

    except Exception as e:
        state.message_init = f"Erreur d'initialisation : {e}"
        print(f"[ERREUR] {e}")
        state.pret = False
    finally:
        state.en_cours = False


def initialiser_rag_background(forcer: bool = False):
    """Lance la construction du RAG dans un thread séparé."""
    thread = threading.Thread(target=_construire_rag, args=(forcer,), daemon=True)
    thread.start()

# ─────────────────────────────────────────────────────────────
# APPLICATION FASTAPI
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialiser_rag_background(forcer=False)
    yield

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
    question:    str
    mode_force:  str = "AUTO"

class MessageResponse(BaseModel):
    reponse: str
    mode:    str
    sources: List[str] = []

# ─────────────────────────────────────────────────────────────
# ROUTES PRINCIPALES
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("templates/index.html")

@app.get("/health")
async def health():
    return {
        "status":    "ok"            if state.pret     else
                     "construction"  if state.en_cours  else
                     "attente",
        "message":   state.message_init,
        "nb_chunks": state.nb_chunks,
    }

@app.post("/chat", response_model=MessageResponse)
async def chat(req: MessageRequest):
    if not state.pret:
        detail = (
            "Le RAG est en cours de construction, veuillez patienter."
            if state.en_cours else
            "Le RAG n'est pas initialisé. Uploadez vos documents via /admin."
        )
        raise HTTPException(status_code=503, detail=detail)

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    mode = req.mode_force.upper()
    if mode == "AUTO":
        try:
            mode_brut = state.chaine_routeur.invoke(question).strip().upper()
            mode = "RAG" if "RAG" in mode_brut else "WEB" if "WEB" in mode_brut else "LLM"
        except Exception:
            mode = "RAG"

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

# ─────────────────────────────────────────────────────────────
# ROUTES D'ADMINISTRATION (upload de documents)
# ─────────────────────────────────────────────────────────────

@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """Page d'administration pour uploader les documents."""
    fichiers = []
    if os.path.isdir(DOCS_DIR):
        # Lister récursivement tous les fichiers supportés avec leur chemin relatif
        for racine, sous_dossiers, noms in os.walk(DOCS_DIR):
            sous_dossiers.sort()
            for nom in sorted(noms):
                if nom.lower().endswith(EXTENSIONS_SUPPORTEES):
                    chemin_relatif = os.path.relpath(os.path.join(racine, nom), DOCS_DIR)
                    fichiers.append(chemin_relatif)

    statut_rag = (
        f"<span style='color:#10b981'>✅ Prêt — {state.nb_chunks} chunks indexés</span>"
        if state.pret else
        "<span style='color:#f59e0b'>⏳ Construction en cours...</span>"
        if state.en_cours else
        f"<span style='color:#ef4444'>❌ Non initialisé — {state.message_init}</span>"
    )

    liste_fichiers = "".join(
        f"<li style='display:flex;justify-content:space-between;align-items:center;"
        f"padding:6px 0;border-bottom:1px solid #eee'>"
        f"<span>📄 {f}</span>"
        f"<button onclick=\"supprimerFichier('{f}')\" "
        f"style='background:#ef4444;color:#fff;border:none;border-radius:4px;"
        f"padding:2px 8px;cursor:pointer;font-size:12px'>Supprimer</button>"
        f"</li>"
        for f in fichiers
    ) or "<li style='color:#6b7280'>Aucun document chargé</li>"

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Administration — RAG Recensement</title>
  <style>
    body {{ font-family: "Segoe UI", Arial, sans-serif; background:#f5f6fa;
            color:#1a1a2e; margin:0; padding:0; }}
    header {{ background:#003189; color:#fff; padding:16px 24px;
              display:flex; align-items:center; gap:12px; }}
    header h1 {{ font-size:18px; margin:0; }}
    .container {{ max-width:700px; margin:32px auto; padding:0 16px; }}
    .card {{ background:#fff; border-radius:12px; padding:24px;
             box-shadow:0 2px 12px rgba(0,0,0,0.08); margin-bottom:20px; }}
    h2 {{ font-size:15px; font-weight:700; margin:0 0 16px;
          text-transform:uppercase; letter-spacing:0.5px; color:#6b7280; }}
    .statut {{ padding:12px 16px; border-radius:8px; background:#f0f4ff;
               border:1px solid #c7d2fe; font-size:14px; margin-bottom:16px; }}
    .drop-zone {{ border:2px dashed #c7d2fe; border-radius:8px; padding:32px;
                  text-align:center; color:#6b7280; cursor:pointer;
                  transition:all 0.2s; background:#fafbff; }}
    .drop-zone:hover, .drop-zone.survol {{ border-color:#003189;
                                           background:#e8edf8; color:#003189; }}
    .drop-zone input {{ display:none; }}
    .btn {{ display:inline-block; padding:10px 20px; border-radius:8px;
            border:none; cursor:pointer; font-size:14px; font-weight:600;
            transition:all 0.15s; }}
    .btn-primary {{ background:#003189; color:#fff; }}
    .btn-primary:hover {{ background:#002070; }}
    .btn-danger  {{ background:#ef4444; color:#fff; }}
    .btn-danger:hover  {{ background:#dc2626; }}
    .btn-success {{ background:#10b981; color:#fff; }}
    .btn-success:hover {{ background:#059669; }}
    ul {{ list-style:none; padding:0; margin:0; }}
    #log {{ background:#1a1a2e; color:#a3e635; font-family:monospace;
            font-size:12px; padding:12px; border-radius:8px;
            min-height:60px; max-height:200px; overflow-y:auto;
            white-space:pre-wrap; display:none; margin-top:12px; }}
    .actions {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:12px; }}
    a.retour {{ color:#003189; text-decoration:none; font-size:13px; }}
    a.retour:hover {{ text-decoration:underline; }}
  </style>
</head>
<body>
<header>
  <div style="width:4px;height:32px;background:linear-gradient(to bottom,#002395 33%,#fff 33%,#fff 66%,#e1000f 66%);border-radius:2px"></div>
  <h1>Administration — RAG Recensement</h1>
  <a href="/" style="margin-left:auto;color:#fff;font-size:13px">← Retour au chat</a>
</header>

<div class="container">

  <!-- Statut -->
  <div class="card">
    <h2>Statut du RAG</h2>
    <div class="statut">{statut_rag}</div>
    <div class="actions">
      <button class="btn btn-success" onclick="reconstruire()">🔄 Reconstruire le RAG</button>
    </div>
    <div id="log"></div>
  </div>

  <!-- Upload -->
  <div class="card">
    <h2>Uploader des documents</h2>
    <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()"
         ondragover="event.preventDefault();this.classList.add('survol')"
         ondragleave="this.classList.remove('survol')"
         ondrop="gererDrop(event)">
      <div style="font-size:36px;margin-bottom:8px">📂</div>
      <div style="font-size:15px;font-weight:600;margin-bottom:4px">Cliquez ou glissez vos fichiers ici</div>
      <div style="font-size:12px">Formats acceptés : PDF, ODP, ODT, XLS, XLSX, ODS</div>
      <input type="file" id="fileInput" multiple accept=".pdf,.odp,.odt,.xls,.xlsx,.ods" onchange="uploaderFichiers(this.files)">
    </div>
    <div id="progression" style="margin-top:12px;font-size:13px;color:#6b7280"></div>
  </div>

  <!-- Liste des documents -->
  <div class="card">
    <h2>Documents actuellement chargés ({len(fichiers)} fichier(s))</h2>
    <ul id="listeFichiers">{liste_fichiers}</ul>
  </div>

</div>

<script>
  async function uploaderFichiers(files) {{
    if (!files || files.length === 0) return;
    const prog = document.getElementById('progression');
    prog.textContent = `Envoi de ${{files.length}} fichier(s)...`;

    const form = new FormData();
    for (const f of files) form.append('files', f);

    try {{
      const r = await fetch('/admin/upload', {{ method: 'POST', body: form }});
      const data = await r.json();
      if (r.ok) {{
        prog.innerHTML = `<span style="color:#10b981">✅ ${{data.message}}</span>`;
        setTimeout(() => location.reload(), 1500);
      }} else {{
        prog.innerHTML = `<span style="color:#ef4444">❌ ${{data.detail || 'Erreur'}}</span>`;
      }}
    }} catch(e) {{
      prog.innerHTML = `<span style="color:#ef4444">❌ Erreur réseau</span>`;
    }}
  }}

  function gererDrop(e) {{
    e.preventDefault();
    document.getElementById('dropZone').classList.remove('survol');
    uploaderFichiers(e.dataTransfer.files);
  }}

  async function supprimerFichier(nom) {{
    if (!confirm(`Supprimer "${{nom}}" ?`)) return;
    const r = await fetch('/admin/supprimer', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{ nom }})
    }});
    const data = await r.json();
    if (r.ok) location.reload();
    else alert(data.detail || 'Erreur');
  }}

  async function reconstruire() {{
    const log = document.getElementById('log');
    log.style.display = 'block';
    log.textContent = 'Lancement de la reconstruction...\\n';
    const r = await fetch('/admin/reconstruire', {{ method: 'POST' }});
    const data = await r.json();
    log.textContent += data.message + '\\n';

    // Polling du statut toutes les 5 secondes
    const poll = setInterval(async () => {{
      try {{
        const s = await fetch('/health');
        const sd = await s.json();
        log.textContent += `Statut : ${{sd.status}} — ${{sd.message || (sd.nb_chunks ? sd.nb_chunks + ' chunks' : '')}}\n`;
        log.scrollTop = log.scrollHeight;
        if (sd.status === 'ok') {{
          clearInterval(poll);
          log.textContent += '✅ Reconstruction terminée !\n';
          setTimeout(() => location.reload(), 1500);
        }} else if (sd.status === 'attente') {{
          // Le RAG n'est plus en construction : soit erreur, soit pas de docs
          clearInterval(poll);
          if (sd.message && sd.message.includes('Erreur')) {{
            log.textContent += `❌ Erreur : ${{sd.message}}\n`;
          }} else if (sd.message) {{
            log.textContent += `⚠️ Arrêt : ${{sd.message}}\n`;
          }} else {{
            log.textContent += '⚠️ Construction terminée avec statut inconnu.\n';
          }}
          setTimeout(() => location.reload(), 2000);
        }}
      }} catch(e) {{
        log.textContent += 'Erreur réseau, nouvelle tentative...\n';
      }}
    }}, 5000);
  }}
</script>
</body>
</html>"""
    return HTMLResponse(html)


@app.post("/admin/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Reçoit des fichiers et les sauvegarde dans DOCS_DIR."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    sauvegardes = []
    extensions_valides = {ext.lstrip('.') for ext in EXTENSIONS_SUPPORTEES}
    for f in files:
        nom = f.filename or "fichier_inconnu"
        ext = nom.lower().rsplit(".", 1)[-1] if "." in nom else ""
        if ext not in extensions_valides:
            continue
        dest = os.path.join(DOCS_DIR, nom)
        contenu = await f.read()
        with open(dest, "wb") as out:
            out.write(contenu)
        sauvegardes.append(nom)
        print(f"[UPLOAD] Sauvegardé : {dest} ({len(contenu)} octets)")

    if not sauvegardes:
        raise HTTPException(
            status_code=400,
            detail=f"Aucun fichier valide reçu. Formats acceptés : {', '.join(sorted(extensions_valides)).upper()}"
        )

    # Lancer la reconstruction du RAG en arrière-plan
    initialiser_rag_background(forcer=True)

    return JSONResponse({
        "message": f"{len(sauvegardes)} fichier(s) uploadé(s) : {', '.join(sauvegardes)}. "
                   f"Reconstruction du RAG lancée en arrière-plan."
    })


@app.post("/admin/supprimer")
async def supprimer_document(payload: dict):
    """Supprime un document du dossier DOCS_DIR (supporte les chemins relatifs avec sous-dossiers)."""
    nom = payload.get("nom", "")
    if not nom:
        raise HTTPException(status_code=400, detail="Nom de fichier invalide.")
    # Sécurité : empêcher les path traversal (../)
    chemin = os.path.normpath(os.path.join(DOCS_DIR, nom))
    if not chemin.startswith(os.path.normpath(DOCS_DIR)):
        raise HTTPException(status_code=400, detail="Chemin de fichier non autorisé.")
    if not os.path.exists(chemin):
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    os.remove(chemin)
    print(f"[ADMIN] Supprimé : {chemin}")
    return JSONResponse({"message": f"{nom} supprimé."})


@app.post("/admin/reconstruire")
async def reconstruire_rag():
    """Relance la construction du RAG depuis les documents présents."""
    if state.en_cours:
        return JSONResponse({"message": "Construction déjà en cours, patientez."})
    initialiser_rag_background(forcer=True)
    return JSONResponse({"message": "Reconstruction du RAG lancée en arrière-plan."})


# ─────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
