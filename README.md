# CS-30-1 Baseline RAG for Personalized AI Learning Assistant

Baseline architecture for a personalised AI learning assistant using RAG and LLM, evaluated on SciQ multiple-choice questions.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. Knowledge Base Build                                             │
│    OpenStax Textbooks → Document Parsing → Structure-aware Chunking │
│    → Dense Embedding (BGE / E5 / GTE) → Vector Store (FAISS /       │
│      pgvector)                                                      │
├─────────────────────────────────────────────────────────────────────┤
│ 2. Online Retrieval & Generation                                    │
│    SciQ Questions                                                   │
│    ├── Path A (E1): Basic Dense RAG                                 │
│    │   Dense Retrieval Top-k → Retrieved Context → Prompt Builder   │
│    │   → Evidence-constrained Prompt → LLM Adapter → Structured     │
│    │   Answer                                                       │
│    └── Path B (E0): Pure LLM Baseline                               │
│        → LLM Adapter → Structured Answer                            │
├─────────────────────────────────────────────────────────────────────┤
│ 3. Evaluation Layer                                                 │
│    E0 & E1 → SciQ MCQ Accuracy → Validate RAG > Pure LLM            │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
5703/
├── config/                 # Configuration files
├── data/                   # Raw and processed data
├── src/
│   ├── knowledge_base/     # Document parsing, chunking, embedding, vector store
│   ├── retrieval_generation/  # Retrieval, prompt building, LLM adapter
│   └── evaluation/         # SciQ evaluation harness
├── scripts/                # End-to-end execution scripts
└── tests/                  # Unit tests
```

## Quick Start

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build the knowledge base from OpenStax textbooks
python scripts/build_knowledge_base.py --config config/config.yaml

# Run retrieval & generation on SciQ questions
python scripts/run_retrieval_generation.py --config config/config.yaml --mode e1

# Evaluate both baselines
python scripts/run_evaluation.py --config config/config.yaml
```

## Baselines

- **E0 (Pure LLM Baseline)**: No retrieval; the LLM answers SciQ questions using only its parametric knowledge.
- **E1 (Basic Dense RAG)**: Retrieves top-k chunks with dense embeddings and constrains the answer with retrieved evidence.

## Primary Metric

- **SciQ MCQ Accuracy**: Exact-match accuracy on the SciQ multiple-choice science questions dataset.
