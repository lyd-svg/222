import chromadb
from research_system.config.settings import CHROMA_PERSIST_DIR
from research_system.tools.embeddings import MultilingualEmbeddingFunction

c = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
col = c.get_collection("latex_papers", embedding_function=MultilingualEmbeddingFunction())
r = col.get(limit=5, include=["metadatas"])
for m in r["metadatas"]:
    tc = m.get("token_count", "MISSING")
    print(f"  token_count={tc}  |  {m['hierarchy'][:50]}")
