# Evaluation / QA Prototype

This directory contains an independent prototype for the Evaluation and Quality Assurance component of the CS-30-1 project.

The prototype is separated from the production source code so that the evaluation design, schemas, metrics, tests, and supporting artifacts can be reviewed independently before integration.

## Current Components

- Fixed evaluation test-set structure
- Evaluation record schema
- SciQ answer-choice handling
- Answer-level evaluation metrics
- Retrieval evaluation metrics
- Unit tests
- Technical documentation

## Directory Structure

```text
evaluation_qa/
├── src/
├── tests/
├── data/
└── docs/

This prototype does not modify the existing production implementation under the repository's
main src/, scripts/, tests/, or config/ directories.
