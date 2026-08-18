# Testing Methodology & Failure Analysis

This document outlines the testing framework, data sample sizes, error categorization, and unresolved issues for the Traffic Law RAG system.

## 1. Test Methodology

The testing approach for the RAG system involves batch-processing diverse sets of real-world user queries (extracted from historical records and test sets) and evaluating the LLM's responses.

### Process
1. **Query Ingestion**: Queries are loaded from various sources (Excel, TXT, JSON).
2. **Batch Execution**: Scripts (like `test_batch_queries.py` and `test_batch_v6.py`) feed these queries into the core RAG pipeline (`rag.py`) asynchronously to simulate production loads.
3. **Retrieval**: The system queries the Qdrant vector database to retrieve the relevant sections of the Motor Vehicles Act (1988) and subsequent amendments (2019).
4. **Evaluation**: Outputs are written to JSON files (e.g., `batch_query_v6_results.json`) and reviewed manually and systematically to identify hallucinations, omissions, or incorrect statutory references.
5. **Multilingual Normalization**: A specifically tailored testing suite ensures that colloquial Hindi/Hinglish terms (e.g., "चालान", "कागज पूरे हैं") are properly normalized before vector search.

## 2. Sample Size

Testing was performed across multiple batches representing different categories of failure modes and edge cases. The current sample size includes:

| Dataset / File | Query Count | Focus Area |
| :--- | :--- | :--- |
| `testcases_txt_results_v2.json` | 31 | General English & Hindi test cases extracted from unstructured text. |
| `batch_query_v6_results.json` | 22 | Specific Hindi/Hinglish failures and colloquial slang. |
| `excel_test_results.json` | 15 | Structured edge-cases from QA spreadsheets. |
| **Total Validated Cases** | **68** | Comprehensive baseline coverage across all categories. |

## 3. Error Categories (Resolved)

Through iterative testing, the following primary error categories were identified and resolved via prompt engineering and retrieval tuning:

### A. Primary Provision Mismatches
* **Issue**: The LLM frequently cited general penalty sections (like Sec 177) or secondary offenses instead of the specific primary statutory provision governing the actual offence (e.g., citing 177 instead of 194D for helmet violations).
* **Resolution**: Implemented the **Primary Provision Rule** in `rag.py`, explicitly instructing the LLM to identify the specific governing rule over general penalties.

### B. Criminal Law Leakage (BNS/IPC)
* **Issue**: For minor traffic infractions, the LLM was incorrectly pulling in severe Indian Penal Code (IPC) or Bharatiya Nyaya Sanhita (BNS) codes (like citing grievous hurt for simple collisions).
* **Resolution**: Added strict boundaries in the system prompt to limit criminal-law content exclusively to evidence protection, preservation, and documentation, keeping the focus strictly on traffic enforcement.

### C. Hindi / Hinglish Tokenization Failures
* **Issue**: The semantic search struggled to map regional slang to legal jargon. For example, a user typing "लाइसेंस घर पर है" (license is at home) failed to map to "license not produced".
* **Resolution**: Introduced a normalization map in the prompt prior to retrieval, ensuring terms like "चालान" map to "challan/enforcement" and "कागज पूरे हैं" map to "documents claimed valid — verify".

## 4. Unresolved Failure Cases

While the vast majority of test cases now pass, the following edge cases remain challenging and require further refinement:

1. **Compound State-Specific Overrides**:
   - The RAG system sometimes struggles when a state (like Chhattisgarh) has a specific compounded fine that explicitly overrides the standard Motor Vehicles Act (2019) penalty. If the context chunk lacks the state-specific modifier, the LLM defaults to the central law.

2. **Complex Multi-Violation Intersections**:
   - In scenarios where a user describes 3-4 distinct violations in a single breathless sentence (e.g., "driver was drunk, had no helmet, no insurance, and tried to run away"), the RAG retrieval occasionally drops one of the lesser charges (like insurance) due to context limits prioritizing the severe offenses (drunk driving, fleeing).

3. **Ambiguous "Other" Clauses**:
   - Queries involving extremely rare edge cases (e.g., specific agricultural vehicle exemptions) occasionally result in the LLM defaulting to the catch-all Section 177 because the specific agricultural clause isn't retrieved with high enough similarity.
