# Data Set Description

### Fine-Tuning Dataset

- **Source:** 2,400 manually curated Q&A pairs derived from internal HR policy documents
- **Coverage:** Leave policy, benefits, code of conduct, travel policy, expense guidelines
- **Annotation:** Each pair reviewed by two HR subject matter experts

### RAG Knowledge Base

- **Documents:** 340 internal policy PDFs and intranet pages
- **Chunks:** ~12,000 text chunks (512 tokens each, 128 token overlap)
- **Embedding Model:** `text-embedding-3-small` (OpenAI)
