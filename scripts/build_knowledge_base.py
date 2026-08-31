"""Build the knowledge base from OpenStax textbooks."""

import argparse
from pathlib import Path

import yaml

from src.knowledge_base.chunker import StructureAwareChunker
from src.knowledge_base.document_parser import PDFParser
from src.knowledge_base.embedder import DenseEmbedder
from src.knowledge_base.vector_store import FAISSVectorStore


def build_knowledge_base(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kb_config = config["knowledge_base"]
    pdf_dir = Path(kb_config["pdf_dir"])

    parser = PDFParser()
    chunker = StructureAwareChunker(**kb_config["chunking"])
    embedder = DenseEmbedder(**kb_config["embedding"])

    vector_store = FAISSVectorStore(
        dimension=embedder.dimension,
        metric=kb_config["vector_store"]["faiss"]["metric"],
    )

    pdf_files = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}. Please add OpenStax textbooks there.")
        return

    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...")
        documents = list(parser.parse(str(pdf_path)))
        chunks = []
        for doc in documents:
            chunks.extend(list(chunker.chunk(doc)))

        if chunks:
            texts = [c["text"] for c in chunks]
            embeddings = embedder.encode(texts)
            vector_store.add(embeddings, chunks)
            print(f"  Added {len(chunks)} chunks from {pdf_path.name}")

    index_path = kb_config["vector_store"]["faiss"]["index_path"]
    vector_store.save(index_path)
    print(f"Knowledge base saved to {index_path}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Build the knowledge base")
    arg_parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    args = arg_parser.parse_args()
    build_knowledge_base(args.config)
