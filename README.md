# GHG-Protocol 
A fully **agentic** chatbot that helps users navigate the Greenhouse-Gas Protocol universe:  
ask questions, upload data files, and retrieve answers grounded in official guidance or in the files you provide.  
Behind the scenes the app orchestrates several specialised agents (history, knowledge, file and retrieval agents) and a vector-search memory to deliver concise, source-linked answers.

---

## 1  Project Goals
* **Conversational GHG assistant** – natural chat UI powered by Azure OpenAI (GPT-4 mini)  
* **Retrieval-Augmented Generation (RAG)** – queries are matched against a Qdrant vector index of key GHG documents.  
* **File-aware reasoning** – users can drop CSV/XLSX/PDF/images; text is extracted, embedded into the JSON request, and the *File Agent* answers from that content.  
* **Persistent sessions** – every chat is stored under `sessions/<sessionId>.json`; past sessions appear in the sidebar and can be re-loaded instantly.

---

## 2  Architecture & Agents
| Agent | Trigger | Responsibility |
|-------|---------|----------------|
| **Selection Agent** | Every user turn | Routes the query (considering history & file context) to one of the specialist agents below. |
| **History Agent** | route =`history` | Summarises or fetches info strictly from conversation history. |
| **Knowledge Agent** | route =`knowledge` | Answers common GHG questions from baked-in guidance snippets. |
| **File Agent** | route =`file` or on fresh upload | Works exclusively on extracted text of the uploaded file(s). |
| **Qdrant Agent** | default / route =`qdrant` | Retrieves best match from the vector DB and answers only from that document. |

⚙️ **Pipeline**

1. User sends a message (plus optional file).  
2. `process_chatbot_request` builds a JSON payload, updates history, and extracts file text if needed.  
3. **Selection Agent** returns the routing keyword.  
4. The chosen specialised agent produces the answer + source metadata.  
5. Streamlit renders the exchange and saves the chat JSON to disk.

---

## 3  Key Files
| File | Purpose |
|------|---------|
| `streamlit_app.py` | Main UI + agent orchestration. |
| `qdrantcollection.py` | Helper to create/update the Qdrant collection from `raw.json` summaries. |
| `raw.json` | Curated GHG documents (filename, URL, summary) for vector indexing. |
| `sessions/` | Auto-generated folder containing per-session chat histories. |

### 3.1  File-Extraction Helpers
Located at the top of `streamlit_app.py`:

* `extract_csv_text` – plain CSV → text  
* `extract_xlsx_text` – each sheet → CSV string  
* `extract_pdf_text` – PyPDF2 page extraction  
* `extract_image_text` – Tesseract OCR  
* `extract_file_content` – dispatcher returning `List[str]` used in the JSON payload.

---

### 4  Requirements & Installation
```bash

pip install -r requirements.txt
```
### 5 Running the Application
```bash
streamlit run app.py
```
### 6 Populating the Vector Database (Qdrant)
Place document summaries in raw.json (same schema).

Edit credentials in qdrantcollection.py.

Run:
```bash
python qdrant.py
The script embeds each summary (Azure OpenAI) and upserts it into Qdrant with filename & source payloads.
```
### 7 Extending the Bot
* **Add Documents** – append to raw.json and re-run the Qdrant upload script.
* **Custom agents** – create a new agent and update selection_agent routing.
* **Advanced chunking** – split large PDFs into paragraph vectors for finer-grained retrieval.

### 8 Security & Confidential Data
All keys in this repo are placeholders. Never commit real credentials.
Use environment variables or secret-management tools when deploying.


### Acknowledgements & Inspiration
Demonstrates a modern agentic RAG stack inspired by OpenAI function-calling patterns and the lightweight yet powerful Qdrant vector store.

Happy decarbonising! 🌍⚡
