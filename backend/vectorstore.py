import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import hashlib
import os

# Use a legal-domain aware embedding model
# legal-bert is fine-tuned on legal text — much better than generic models
# for Indian legal documents
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"# Note: for production, switch to "nlpaueb/legal-bert-base-uncased"
# but MiniLM is faster and good enough for demo

class LegalVectorStore:
    def __init__(self, persist_dir="./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        self.model.max_seq_length = 256  # reduce from default 384, saves memory

    def get_or_create_collection(self, doc_id: str):
        """Each document gets its own collection."""
        return self.client.get_or_create_collection(
            name=f"legal_{doc_id}",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_document(self, doc_id: str, chunks: list[dict]) -> int:
        """
        Embed chunks and store in ChromaDB.
        
        What's happening here:
        1. Each chunk of text → embedding model → 384-dim vector
        2. Store vector + original text + metadata in ChromaDB
        3. ChromaDB builds an HNSW index for fast approximate nearest-neighbour search
        """
        collection = self.get_or_create_collection(doc_id)
        
        # check if already indexed
        if collection.count() > 0:
            return collection.count()
        
        texts = [c["content"] for c in chunks]
        
        # generate embeddings in batches
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        ).tolist()
        
        # store in chromadb
        collection.add(
            ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{
                "section_idx": c["section_idx"],
                "chunk_idx": c["chunk_idx"],
                **{k: str(v) for k, v in c["metadata"].items()}
            } for c in chunks]
        )
        
        return len(chunks)
    
    def search(self, doc_id: str, query: str, k: int = 5) -> list[dict]:
        """
        Semantic search: find chunks most relevant to the query.
        
        What's happening:
        1. Embed the query using the same model
        2. ChromaDB finds k nearest vectors by cosine similarity
        3. Return the original text chunks + similarity scores
        """
        collection = self.get_or_create_collection(doc_id)
        
        if collection.count() == 0:
            return []
        
        # embed query
        query_embedding = self.model.encode([query]).tolist()
        
        # search
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, collection.count()),
            include=["documents", "metadatas", "distances"]
        )
        
        chunks = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            chunks.append({
                "content": doc,
                "metadata": meta,
                "similarity": round(1 - dist, 4),  # cosine distance → similarity
                "rank": i + 1
            })
        
        return chunks
    
    def document_exists(self, doc_id: str) -> bool:
        try:
            col = self.client.get_collection(f"legal_{doc_id}")
            return col.count() > 0
        except:
            return False
    
    def list_documents(self) -> list[str]:
        collections = self.client.list_collections()
        return [c.name.replace("legal_", "") for c in collections]


def get_doc_id(filename: str, file_bytes: bytes) -> str:
    """Generate stable ID from filename + content hash."""
    content_hash = hashlib.md5(file_bytes).hexdigest()[:8]
    clean_name = filename.replace('.pdf', '').replace(' ', '_')[:30]
    return f"{clean_name}_{content_hash}"