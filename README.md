# Chatbot with RAG

This guide explains end-to-end **RAG pipeline** and how to use the **RAG_chatbot** notebook. This tool is designed to work with the provided data files (`documents.csv`, `single_passage_answer_questions.csv`, `multi_passage_answer_questions.csv`, `no_answer_questions.csv`). The implementation goes beyond basic semantic search by incorporating intelligent data processing, hybrid retrieval strategies, and domain-aware chunking.

## Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or VS Code with Jupyter extension
- Virtual environment (recommended)

---

## Getting Started

### Step 1: Set Up Virtual Environment (Recommended)

It's highly recommended to create a virtual environment to avoid dependency conflicts:

```bash
# Create virtual environment
python -m venv sengpan_rag_module_env

# Activate virtual environment
# On macOS/Linux:
# source sengpan_rag_module_env/bin/activate

# On Windows:
sengpan_rag_module_env\Scripts\activate
```

### Step 2: Install Required Packages

Install all necessary dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```

**Alternative installation (if requirements.txt is not available):**
```bash
pip install jupyter faiss-cpu langchain langchain-community langchain-text-splitters langchain-core langgraph tqdm pandas numpy typing-extensions urllib3 google-generativeai transformers
```

### Step 3: Launch Jupyter Notebook

```bash
jupyter notebook
```

Or if using VS Code, simply open the `.ipynb` files directly.


---

## Data Requirements

### **Required Input Files**
Ensure the following files are present in the project root directory:
- `documents.csv` - 20 documents dataset with mixed topics (game wiki, data-science blog, EU AI Act Q&A, etc.)

Input file should contain at least the following columns:
- `index` : document's index number
- `source_url` : original source of context
- `text` : document's text content

### **Evaluation & Testing Files**
- **`single_passage_answer_questions.csv`**: Contains a set of questions answerable from a single document
- **`multi_passage_answer_questions.csv`**: Contains a set of questions answerable from multi documents
- **`no_answer_questions.csv`**: Contains a set of questions with no answer

Each testing file should contain at least the following columns:
- `document_index`, `question`, `answer` 

---

## RAG Pipeline

### **Core RAG Concept**
RAG connects an LLM to an external database, allowing it to **retrieve** relevant **context** to **generate** more accurate and verifiable answers.

### **Key Architecture**

#### 1. Data Ingestion & Enrichment (`DocumentProcessor`)
**Purpose:** Transforms raw data into a structured, information-rich format optimized for retrieval.

**Features:**
- **Source Classification:** Automatically categorizes sources (e.g., `wiki`, `code-repo`, `docs`).
- **Content Domain Detection:** Identifies content type (`gaming`, `technical`, `narrative`, `dnd`) using keyword analysis.
- **Entity Extraction:** Pulls out key entities, numbers, and quoted terms.
- **Metadata Generation:** Creates titles, previews, and keyword lists for each document.
- **Rich Text Metadata Creation:** Combines metadata and original content into a single, search-enhanced text block.

**Example of Rich Text Metadata:**

`Title`: Cave Giant Encounter Strategies
`Source`: https://gaming-wiki.example/boss-fights (domain: gaming-wiki.example, type: wiki)
`Content` Domain: gaming
`Length`: 450 words, 2450 chars, 15 lines
`Keywords`: giant, cave, damage, resistance, rock-throw, weak-spot
`Key Entities`: Cave Giant, Rock Throw, Ankle Slam, Crystal Weakpoint
`Key Numbers`: 1500 health, 50% blunt resistance
`Preview`: Cave giants are large enemies found in the northern tunnels. They possess high health and are resistant to blunt damage...
`Content`: Cave giants are large enemies found in the northern tunnels. They possess high health (1500 base) and are 50% resistant to blunt weapons. Their primary attacks are Rock Throw and Ankle Slam. A crystal on their back is their weak point, taking double damage...


#### 2. Intelligent Chunking (`MultiStrategyTextSplitter`)

**Purpose:** Breaks documents into meaningful segments without losing critical information.

**Strategy:** Uses different splitting rules based on content domain:
- **Technical:** Splits on markdown headers (`##`, `###`)
- **Narrative:** Splits on chapters
- **D&D/TTRPG:** Splits on narrative breaks (`Day X`, `---`)
- **Gaming:** Balanced chunking for guides and mechanics
- **General:** Fallback strategy for other content

#### 3. Embeddings & Vector Store (Sentence Transformers + FAISS)

**Purpose:** Creates a searchable database that understands semantic meaning.

**Implementation:**
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` converts rich text to numerical vectors
- **Vector Index:** FAISS for fast similarity search
- **Batch Processing:** Efficient ingestion of documents in batches

#### 4. Hybrid Retrieval Engine (`Retriever`)

**Purpose:** Finds the most relevant context for a user's query using multiple strategies.

**Retrieval Strategies:**
1. **Similarity Search:** Finds semantically similar chunks
2. **MMR (Maximal Marginal Relevance):** Balances similarity with diversity
3. **Similarity Threshold:** Only returns results above confidence threshold

**Advanced Features:**
- **Query Analysis:** Extracts key terms, entities, and numbers from queries
- **Custom Scoring:** Re-ranks results based on:
  - Keyword and entity matching
  - Domain and source-type relevance
  - Content quality signals
  - Strategic weighting

#### 5. Generation & Prompt Engineering

**Purpose:** Synthesizes accurate answers from retrieved context.

**Key Features:**
- **Strict Instructions:** LLM is constrained to use only provided context
- **Structured Context:** Clear formatting with source provenance
- **Safety Controls:** Defaults to "I don't know" when context is insufficient
- **Domain Awareness:** Leverages metadata for more nuanced responses

#### 6. LangGraph Workflow

**Purpose:** Manages the multi-step RAG pipeline cleanly and efficiently.

**Pipeline:** Start (user query) -> Retrieve -> Generate answer

**Why it matters:** Provides a robust, scalable framework for complex AI workflows that's easier to maintain and extend.

---

## FULL RAG chatbot (`RAG_chatbot.ipynb`)

### **Purpose**
- Designed for multi-domain knowledge retrieval with enterprise-grade safety controls
- Evaluate with three categories question type (`single_passage_answer_questions.csv`, `multi_passage_answer_questions.csv`, `no_answer_questions.csv`)

### **Performance Results**
1. **Overall Performance**
   - **Total Test Cases**: 12 questions across 3 categories
   - **Overall Accuracy**: 91.7% (11/12 correct responses)
   - **System Reliability**: High consistency across different query types

**Category-specific performance**
- **Multi Passage**: 4/4 (100.0%)
- **No Answer**: 3/4 (75.0%)
- **Single Passage**: 4/4 (100.0%)
 
2. **Safety & Guardrails Testing**
   - Political & Harmful Query Blocking: Successfully blocked inappropriate questions
   - Response: "I don't know" (as designed for blocked content)
   - Safety System Effectiveness: 100% blocking rate for tested harmful patterns

### **Technical Overview**
- Multi-Domain Expertise: Handle diverse content types (gaming, D&D, technical docs, narratives) with domain-specific optimization
- Enterprise Safety: Multi-layer security preventing harmful, ungrounded, or inappropriate responses

1. **Domain-Aware Document Processing**: auto-categorizes content into specialized domains, extracts key entities and terms, and enriches documents with descriptive metadata.
2. **Multi-Strategy Retrieval Engine**: combines multiple retrieval methods, uses custom relevance scoring, and removes duplicates.
3. **Adaptive Text Splitting**: Domain-specific chunking strategries uses context-specific separators and optimized overlap to preserve semantic structure.
4. **Multi-Layer Safety Checking**: 
    - Rule-based pattern matching
    - LLM-based content moderation 
    - Post-generation grounding validation
5. **State Graph Architecture (LangGraph)**
    - Sequential Processing: Retrieve → Generate Answer
    - State Management: Tracks question, context, answer, and debug info throughout the pipeline

Example of LangGraph Usage:
```python
question = "What are effective strategies against cave giants?"
result = graph.invoke({"question": question})
print(result["answer"])
```

### **Chatbot Flow in Detail**
User Query → Safety Check → Query Preprocessing → Multi-Strategy Retrieval → Document Scoring & Ranking → Context Formatting → LLM Generation → Post-Generation Safety → Final Answer

### **How to Use Notebook**
1. Ensure your virtual environment is activated and all dependencies are installed
2. Prepare Data: Place your `documents.csv` file in the `RAG_data/` directory
3. Run all cells sequentially. The notebook will:
   - Load the dataset
   - Embed rich text with `sentence-transformers/all-MiniLM-L6-v2`
   - Faiss with L2 distance indexing (vector store)
   - Use the compiled LangGraph's graph to query your knowledge base
   - Evaluate performance (use `single_passage_answer_questions.csv`, `multi_passage_answer_questions.csv`, `no_answer_questions.csv`)

**Expected Runtime:** 3-4 minutes depending on dataset size.

---

## **Project Structure**

```
your_project/
├── RAG_data/
│   ├── documents.csv
│   ├── single_passage_answer_questions.csv
│   ├── multi_passage_answer_questions.csv 
│   └── no_answer_questions.csv 
├── RAG_chatbot.ipynb
├── documents_metadata.json
├── rich_text.jsonl
└── rich_faiss_index/
    
```

---

## Troubleshooting

### Common Issues and Solutions

1. **ModuleNotFoundError**: Ensure virtual environment is activated and packages are installed
   ```bash
   source sengpan_rag_module_env/bin/activate  # Activate environment
   pip install -r requirements.txt              # Install dependencies
   ```

2. **File not found errors**: Verify that all required `.csv` files are in the project root directory

4. **Memory issues**: For large datasets, consider processing data in smaller chunks or using a machine with more RAM

### Performance Optimization

1. **Retrieval Performance**
- Use FAISS indexing instead of flat search for O(n) complexity on large datasets
- Run retrieval strategies in parallel using threading instead of sequential execution
- Cache frequent queries and retrieval results to avoid redundant searches

2. **Vector Store Optimization**
- Batch embed documents during indexing for 3-5x faster processing
- Use quantized embeddings (8-bit) to reduce memory by 75% with minimal accuracy loss
- Implement lazy loading for document chunks to handle large collections

3. **LLM Generation Optimization**
- Cache LLM responses for identical context+query combinations
- Use streaming responses for better perceived performance
- Truncate context based on relevance scores to fit token limits efficiently

4. **Safety System Optimization**
- Compile regex patterns once at startup for faster rule-based blocking
- Use lightweight models for content moderation instead of full LLMs
- Early stopping in grounding checks when high overlap is detected

---

## Expected Results

### Chatbot Output
- **documents_metadata.json**: Complete processed metadata
- **rich_text.jsonl**: Enriched text chunks for searching
- **rich_faiss_index/**: FAISS vector store for production use

---

## Support

For issues, questions, or contributions:
1. Check the troubleshooting section above
2. Verify all prerequisites are met
3. Ensure data format matches requirements
4. Review notebook comments and markdown cells for detailed explanations

---

**Note**: This RAG pipeline is specifically designed for production-grade performance, achieving 91.7% accuracy across diverse query types. Ensure the multi-layer safety system, domain-aware processing, and hybrid retrieval approach are configured for specific knowledge base to maintain optimal reliability. The comprehensive evaluation framework use to establish baseline metrics before deployment and for optimization efforts.
