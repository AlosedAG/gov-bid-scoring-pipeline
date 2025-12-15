# Bid Opportunity Classification & Scoring Pipeline

This repository contains an end-to-end Python system for identifying, filtering, and prioritizing government bid opportunities with a strict focus on software-related solicitations.

The pipeline was designed to replace manual bid review workflows and prevent irrelevant or low-quality opportunities from entering downstream systems such as CRMs. It combines deterministic business rules with a trained NLP classification model to improve accuracy, transparency, and maintainability.

## Overview

Raw bid data is ingested and normalized, then evaluated across multiple dimensions including location validity, population thresholds, state prioritization, RFx type detection, aggregator source detection, and deadline constraints. Each opportunity is scored, ranked, and labeled with human-readable explanations describing why it was accepted, deprioritized, or discarded.

To ensure observability and data quality, results are written to spreadsheets first, allowing for auditing and iteration before optional integration with external systems.

## Key Features

- Hybrid rule-based and machine learning classification
- NLP model trained to distinguish software vs non-software solicitations
- Location normalization (city vs county) with population thresholds
- State-tier prioritization logic
- Aggregator source detection and exclusion
- RFx type detection (RFI, RFQ, IFB, etc.)
- Deadline validation
- Explainable scoring with per-record reasoning
- Spreadsheet-first output for monitoring and validation
- Designed for safe downstream integration (CRM / automation tools)

## Technology Stack

- Python
- Pandas, NumPy
- scikit-learn / Transformers
- Regular expressions and rule-based filtering
- Google Sheets integration for monitoring
- Joblib for model persistence

- 
## Usage Notes

This repository represents an independently developed implementation. The system is intentionally modular so that scoring logic, thresholds, and model components can be adjusted without impacting the rest of the pipeline.

Model weights may be excluded or replaced depending on usage context.

## Author

Aylín Altamirano  
Initial development: 2025

This project was built as an original implementation for bid analysis, automation, and data quality optimization.
