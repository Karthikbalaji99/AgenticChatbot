# 🌿 GHG-Protocol Assistant

A fully **agentic** 🤖 chatbot that helps users navigate the Greenhouse Gas (GHG) Protocol universe:  
Ask questions, upload 📁 data files, and retrieve answers grounded in official 📚 guidance or in the files you provide.  
Behind the scenes, the app orchestrates several specialised agents 🧠 and a vector-search memory 🔍 to deliver concise, source-linked answers ✅.

---

## 1️⃣ Project Goals
* 🗨️ **Conversational GHG Assistant** – natural chat UI powered by Azure OpenAI (GPT-4 mini)  
* 📦 **Retrieval-Augmented Generation (RAG)** – queries matched against a Qdrant vector index of key GHG documents  
* 📂 **File-aware reasoning** – users can drop CSV/XLSX/PDF/images; text is extracted and reasoned over by the *File Agent*  
* 💾 **Persistent sessions** – chats saved under `sessions/<sessionId>.json`; reload past sessions instantly from the sidebar

---

## 2️⃣ Architecture & Agents
| 🤖 **Agent** | 🔄 **Trigger** | 🧠 **Responsibility** |
|-------------|----------------|------------------------|
| **Selection Agent** | Every user turn | Routes the query (considering history & file context) to a specialist agent below |
| **History Agent** | `route=history` | Summarises or fetches info strictly from chat history 📜 |
| **Knowledge Agent** | `route=knowledge` | Answers general GHG questions from built-in knowledge 🔍 |
| **File Agent** | `route=file` or on file upload | Analyzes uploaded file content (CSV, XLSX, PDF, image) 📄 |
| **Qdrant Agent** | Default / `route=qdrant` | Retrieves most relevant GHG doc match from the vector DB 📖 |

⚙️ **Pipeline**  
1. ✉️ User sends a message (plus optional file)  
2. 🧰 `process_chatbot_request` builds a JSON payload, extracts file text if needed  
3. 🚦 Selection Agent returns a routing keyword  
4. 🧭 Chosen agent produces the answer + source metadata  
5. 🎨 Streamlit renders the response and saves chat to disk

---

## 3️⃣ Key Files
| 📁 File | 📌 Purpose |
|--------|------------|
| `streamlit_app.py` | Main UI + orchestrator for agent logic |
| `qdrantcollection.py` | Creates/updates Qdrant collection from `raw.json` summaries |
| `raw.json` | Curated GHG documents for vector search indexing |
| `sessions/` | Folder containing per-session chat histories 💬 |

### 3.1 🔍 File-Extraction Helpers
Located at the top of `streamlit_app.py`:

* `extract_csv_text` – Extracts plain CSV → text  
* `extract_xlsx_text` – Each sheet → CSV string  
* `extract_pdf_text` – PyPDF2 page extraction 📄  
* `extract_image_text` – OCR from images via Tesseract 🖼️  
* `extract_file_content` – Dispatcher returns `List[str]` for JSON embedding 📦

---

## 4️⃣ Requirements & Installation 🛠️
```bash
pip install -r requirements.txt
```
### 5 Running the Application  🚀
```bash
streamlit run app.py
```
### 6 Populating the Vector Database (Qdrant) 🧠
Place document summaries in raw.json (same schema).

Edit credentials in qdrantcollection.py.

Run:
```bash
python qdrant.py
The script embeds each summary (Azure OpenAI) and upserts it into Qdrant with filename & source payloads.
```
### 7 Extending the Bot 🧩
* **📥 Add Documents** – append to raw.json and re-run the Qdrant upload script.
* **🧑‍💻 Custom agents** – create a new agent and update selection_agent routing.
* **✂️ Advanced chunking** – split large PDFs into paragraph vectors for finer-grained retrieval.

### 8 Security & Confidential Data 🔐
❗All keys in this repo are placeholders. Never commit real credentials.
Use environment variables or secret-management tools when deploying.


### Acknowledgements & Inspiration 🙌
Demonstrates a modern agentic RAG stack 🧠✨ inspired by OpenAI function-calling patterns and the lightweight yet powerful Qdrant vector store🗂️.

Happy decarbonising! 🌍⚡
