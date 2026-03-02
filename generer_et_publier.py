"""
=============================================================
  GÉNÉRATEUR D'INDEX FAISS + PUBLICATION AUTOMATIQUE GITHUB
=============================================================
  Ce script :
    1. Génère l'index FAISS depuis vos documents locaux
    2. Pousse automatiquement les fichiers sur GitHub via l'API
       (fonctionne même pour des fichiers > 25 Mo)

  CONFIGURATION (à modifier une seule fois ci-dessous) :
    - ALBERT_API_KEY  : votre clé API Albert
    - GITHUB_TOKEN    : votre token GitHub (droits "repo")
    - GITHUB_REPO     : votre dépôt GitHub (propriétaire/nom)

  USAGE :
    Double-cliquez sur "generer_et_publier.bat"
    ou lancez : python generer_et_publier.py
=============================================================
"""

import os, sys, shutil, base64, requests
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from lxml import etree

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION — À MODIFIER ICI
# ══════════════════════════════════════════════════════════════
ALBERT_API_KEY  = os.getenv("ALBERT_API_KEY",  "sk-eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjo5MzMzLCJ0b2tlbl9pZCI6MTUwOTIsImV4cGlyZXMiOjE3OTg2NzE2MDB9.IrqDOCL-mq2sgvcUuj8VAlzEtynZKWuBpvo2idfS1ak")
ALBERT_BASE_URL = os.getenv("ALBERT_BASE_URL", "https://albert.api.etalab.gouv.fr/v1")
EMBED_MODEL     = os.getenv("EMBED_MODEL",     "BAAI/bge-m3")
DOCS_DIR        = os.getenv("DOCS_DIR",        "./documents")
FAISS_INDEX     = os.getenv("FAISS_INDEX",     "faiss_index_recensement")

GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN",    "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")
GITHUB_REPO     = os.getenv("GITHUB_REPO",     "VincentB1984/test-rag-rp")
GITHUB_BRANCH   = os.getenv("GITHUB_BRANCH",   "main")
# ══════════════════════════════════════════════════════════════


# ── Classe d'embeddings personnalisée ──────────────────────────
class AlbertEmbeddings(Embeddings):
    def __init__(self, api_key, base_url, model):
        self._api_key  = api_key
        self._base_url = base_url.rstrip("/")
        self._model    = model
        self._headers  = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _embed(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            nb_lots = (len(texts) - 1) // batch_size + 1
            print(f"  Vectorisation lot {i//batch_size + 1}/{nb_lots} ({len(batch)} textes)...")
            payload = {
                "model": self._model,
                "input": batch,
                "encoding_format": "float",
            }
            resp = requests.post(
                f"{self._base_url}/embeddings",
                headers=self._headers,
                json=payload,
                timeout=120,
            )
            if not resp.ok:
                raise RuntimeError(f"API error {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            items = sorted(data["data"], key=lambda x: x["index"])
            all_embeddings.extend(item["embedding"] for item in items)
        return all_embeddings

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        return self._embed([text])[0]


# ── Extraction ODP ──────────────────────────────────────────────
def extraire_odp(chemin: str) -> List[Document]:
    docs = []
    try:
        import zipfile
        with zipfile.ZipFile(chemin, "r") as z:
            if "content.xml" not in z.namelist():
                return docs
            with z.open("content.xml") as f:
                tree = etree.parse(f)
        ns = {
            "draw": "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0",
            "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
        }
        pages = tree.findall(".//draw:page", ns)
        for i, page in enumerate(pages, 1):
            textes = [t.strip() for t in page.itertext() if t.strip()]
            contenu = " ".join(textes)
            if contenu:
                docs.append(Document(
                    page_content=contenu,
                    metadata={"source": os.path.basename(chemin), "slide": i, "type": "odp"}
                ))
    except Exception as e:
        print(f"  ⚠️  Erreur ODP {chemin}: {e}")
    return docs


# ── Chargement des documents ────────────────────────────────────
def charger_dossier(dossier: str) -> List[Document]:
    docs = []
    if not os.path.isdir(dossier):
        print(f"❌ Dossier introuvable : {dossier}")
        return docs
    fichiers = [f for f in sorted(os.listdir(dossier))
                if f.lower().endswith((".pdf", ".odp"))]
    if not fichiers:
        print(f"❌ Aucun fichier PDF ou ODP dans {dossier}")
        return docs
    for fichier in fichiers:
        chemin = os.path.join(dossier, fichier)
        print(f"  📄 {fichier}")
        if fichier.lower().endswith(".pdf"):
            try:
                loader = PyPDFLoader(chemin)
                pages = loader.load()
                for p in pages:
                    p.metadata["source"] = fichier
                docs.extend(pages)
            except Exception as e:
                print(f"  ⚠️  Erreur PDF {fichier}: {e}")
        elif fichier.lower().endswith(".odp"):
            docs.extend(extraire_odp(chemin))
    return docs

# ── Publication sur GitHub ──────────────────────────────
GITHUB_CHUNK_SIZE = 20 * 1024 * 1024  # 20 Mo par morceau

def github_get_sha(path):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json().get("sha")
    return None

def github_delete(remote_path, message):
    """Supprime un fichier sur GitHub s'il existe."""
    sha = github_get_sha(remote_path)
    if not sha:
        return  # fichier inexistant, rien à faire
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{remote_path}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    payload = {"message": message, "sha": sha, "branch": GITHUB_BRANCH}
    requests.delete(url, headers=headers, json=payload, timeout=30)

def github_upload_raw(content_bytes, remote_path, message):
    """Uploade des bytes bruts sur GitHub."""
    size_mo = len(content_bytes) / 1024 / 1024
    print(f"  📤 Upload : {remote_path} ({size_mo:.1f} Mo)...")
    content_b64 = base64.b64encode(content_bytes).decode()
    sha = github_get_sha(remote_path)
    payload = {
        "message": message,
        "content": content_b64,
        "branch": GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{remote_path}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    r = requests.put(url, headers=headers, json=payload, timeout=180)
    if r.status_code in (200, 201):
        print(f"  ✅ {remote_path} → publié avec succès")
        return True
    else:
        print(f"  ❌ Erreur {r.status_code} : {r.text[:300]}")
        return False

def github_upload(local_path, remote_base, message):
    """Uploade un fichier en le découpant en morceaux si nécessaire."""
    with open(local_path, "rb") as f:
        data = f.read()
    total = len(data)

    if total <= GITHUB_CHUNK_SIZE:
        # Fichier assez petit : upload direct
        # Supprimer d'éventuels anciens morceaux
        for i in range(10):
            github_delete(f"{remote_base}.part{i:02d}", message)
        return github_upload_raw(data, remote_base, message)
    else:
        # Fichier trop grand : découpage en morceaux
        # Supprimer l'éventuel ancien fichier non découpé
        github_delete(remote_base, message)
        chunks = []
        for i in range(0, total, GITHUB_CHUNK_SIZE):
            chunks.append(data[i:i + GITHUB_CHUNK_SIZE])
        nb = len(chunks)
        print(f"  ℹ️  Fichier de {total/1024/1024:.1f} Mo découpé en {nb} morceaux")
        # Supprimer les anciens morceaux en surplus
        for i in range(nb, nb + 5):
            github_delete(f"{remote_base}.part{i:02d}", message)
        # Uploader les morceaux
        ok = True
        for i, chunk in enumerate(chunks):
            ok = ok and github_upload_raw(chunk, f"{remote_base}.part{i:02d}", message)
        # Écrire un fichier manifeste indiquant le nombre de morceaux
        manifest = f"{nb}".encode()
        ok = ok and github_upload_raw(manifest, f"{remote_base}.manifest", message)
        return ok


# ── Programme principal ─────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  GÉNÉRATION + PUBLICATION DE L'INDEX FAISS")
    print("=" * 60)

    # Vérification de la clé Albert
    if not ALBERT_API_KEY or ALBERT_API_KEY.startswith("VOTRE"):
        print("❌ ALBERT_API_KEY non configurée. Modifiez le script.")
        input("Appuyez sur Entrée pour quitter...")
        sys.exit(1)

    # Vérification du token GitHub
    if not GITHUB_TOKEN or GITHUB_TOKEN.startswith("ghp_VOTRE"):
        print("❌ GITHUB_TOKEN non configuré. Modifiez le script.")
        input("Appuyez sur Entrée pour quitter...")
        sys.exit(1)

    # ── Étape 1 : Chargement des documents
    print(f"\n[1/4] Chargement des documents depuis : {DOCS_DIR}")
    documents = charger_dossier(DOCS_DIR)
    if not documents:
        print("❌ Aucun document chargé. Vérifiez le dossier 'documents\'.")
        input("Appuyez sur Entrée pour quitter...")
        sys.exit(1)
    print(f"  → {len(documents)} pages/slides chargées")

    # ── Étape 2 : Découpage
    print(f"\n[2/4] Découpage en chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    print(f"  → {len(chunks)} chunks créés")

    # ── Étape 3 : Vectorisation
    print(f"\n[3/4] Vectorisation (peut prendre plusieurs minutes)...")
    embeddings = AlbertEmbeddings(
        api_key=ALBERT_API_KEY,
        base_url=ALBERT_BASE_URL,
        model=EMBED_MODEL,
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Sauvegarde locale
    if os.path.exists(FAISS_INDEX):
        shutil.rmtree(FAISS_INDEX)
    vectorstore.save_local(FAISS_INDEX)
    print(f"  → Index sauvegardé localement dans '{FAISS_INDEX}/'")

    # ── Étape 4 : Publication sur GitHub
    print(f"\n[4/4] Publication sur GitHub ({GITHUB_REPO})...")
    commit_msg = f"Mise à jour index FAISS ({len(chunks)} chunks, {len(documents)} pages)"
    ok1 = github_upload(
        os.path.join(FAISS_INDEX, "index.faiss"),
        f"{FAISS_INDEX}/index.faiss",
        commit_msg,
    )
    ok2 = github_upload(
        os.path.join(FAISS_INDEX, "index.pkl"),
        f"{FAISS_INDEX}/index.pkl",
        commit_msg,
    )

    print()
    print("=" * 60)
    if ok1 and ok2:
        print(f"  ✅ Terminé ! Index publié sur GitHub.")
        print(f"  → {len(chunks)} chunks | {len(documents)} pages/slides")
        print(f"  → Render va redéployer automatiquement.")
        print(f"  → Le RAG sera prêt en quelques secondes au prochain démarrage.")
    else:
        print("  ⚠️  L'index a été généré localement mais la publication")
        print("      sur GitHub a échoué. Vérifiez votre GITHUB_TOKEN.")
    print("=" * 60)
    print()
    input("Appuyez sur Entrée pour fermer...")
