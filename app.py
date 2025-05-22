import streamlit as st
import os
import uuid
import json
import asyncio
import time
import logging
from io import BytesIO
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from docx import Document
from qdrant_client import QdrantClient
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient
from datetime import datetime
import pandas as pd
import PyPDF2
from PIL import Image
import pytesseract
import io

def extract_csv_text(uploaded_file) -> str:
    # CSV is plain text
    return uploaded_file.getvalue().decode('utf-8', errors='ignore')

def extract_xlsx_text(uploaded_file) -> str:
    # Read all sheets and convert to CSV-format text
    output = []
    excel_data = pd.read_excel(uploaded_file, sheet_name=None)
    for sheet_name, df in excel_data.items():
        output.append(f"--- Sheet: {sheet_name} ---")
        output.append(df.to_csv(index=False))
    return "\n".join(output)

def extract_pdf_text(uploaded_file) -> str:
    # Read each page with PyPDF2
    reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)

def extract_image_text(uploaded_file) -> str:
    # Run OCR via Tesseract
    img = Image.open(io.BytesIO(uploaded_file.getvalue()))
    return pytesseract.image_to_string(img)

def extract_file_content(uploaded_file) -> List[str]:
    """
    Dispatch based on MIME type / extension.
    Returns a list of text blocks (or a single-item list).
    """
    content_type = uploaded_file.type
    if uploaded_file.name.lower().endswith(".csv"):
        text = extract_csv_text(uploaded_file)
    elif uploaded_file.name.lower().endswith((".xls", ".xlsx")):
        text = extract_xlsx_text(uploaded_file)
    elif uploaded_file.name.lower().endswith(".pdf"):
        text = extract_pdf_text(uploaded_file)
    elif uploaded_file.name.lower().endswith((".png", ".jpg", ".jpeg")):
        text = extract_image_text(uploaded_file)
    else:
        raise ValueError("Unsupported file type")
    # split into reasonable chunks if too large, or return as single block
    return [text]

# ---------------------------------------------------------------------
AZURE_ENDPOINT = "YOUR_AZURE_ENDPOINT"
AZURE_API_KEY = "your_azure_api_key"
OPENAI_API_VERSION = "your_openai_api_version"
# Qdrant Collection Info    


QDRANT_URL = "your_qdrant_url"
QDRANT_API_KEY = "your_qdrant_api_key"
QDRANT_COLLECTION = "your_qdrant_collection_name"

BLOB_CONNECTION_STRING = (
    "your_blob_connection_string"
)
BLOB_CONTAINER = "your_blob_container_name"


SESSION_DIR = "sessions"
TRUNC = 200

# ---------------------- Setup ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
if not os.path.exists(SESSION_DIR):
    os.makedirs(SESSION_DIR)

SEL_SYS_SIMPLE = """
You are a specialized routing assistant for a GHG Protocol chatbot system. Your sole purpose is to analyze user queries and route them to the appropriate specialized agent.

INSTRUCTIONS:
- Analyze the user query to determine its intent and type
- Respond with ONLY ONE of these exact routing keywords:
  - 'history' - if the user is asking about past conversations or message history
  - 'qdrant' - for specific technical questions that likely require document retrieval
  - 'knowledge' - for common general questions about GHG Protocol that don't require specific document lookup

CONSTRAINTS:
- Never explain your reasoning or provide any additional text
- Never attempt to answer the user's question directly
- Always respond with exactly one word: either 'history', 'qdrant', or 'knowledge'
- If uncertain, default to 'qdrant'

EXAMPLES:
User: "What did I ask earlier about scope 3 emissions?"
Response: history

User: "How do I calculate my scope 2 emissions using the market-based method?"
Response: qdrant

User: "Can you summarize our conversation?"
Response: history

User: "What is the GHG Protocol standard for biogenic emissions in my specific industry?"
Response: qdrant

User: "How often should I report my emissions?"
Response: knowledge

User: "What are the three scopes of emissions?"
Response: knowledge

User: "What is the difference between location-based and market-based methods?"
Response: knowledge
"""

# Replace SEL_SYS_EXTENDED with:
SEL_SYS_EXTENDED = """
You are a specialized routing assistant for a GHG Protocol chatbot system. Your sole purpose is to analyze user queries and route them to the appropriate specialized agent.

INSTRUCTIONS:
- Analyze the user query to determine its intent and type
- Consider if the query might be referring to previously uploaded files in the conversation history
- Respond with ONLY ONE of these exact routing keywords:
  - 'history' - if the user is asking about past conversations or message history
  - 'qdrant' - for queries related to GHG Protocol information that require specific document retrieval
  - 'knowledge' - for common general questions about GHG Protocol that don't require specific document lookup
  - 'file' - if the user is likely referring to content from previously uploaded files

CONSTRAINTS:
- Never explain your reasoning or provide any additional text
- Never attempt to answer the user's question directly
- Always respond with exactly one word: either 'history', 'qdrant', 'knowledge', or 'file'
- If uncertain about file references, default to other appropriate categories
- If uncertain between knowledge and qdrant, consider if this is common knowledge (knowledge) or requires specific technical details (qdrant)

EXAMPLES:
User: "What did I ask earlier about scope 3 emissions?"
Response: history

User: "How do I calculate my scope 2 emissions according to the latest standard?"
Response: qdrant

User: "How often should I report my GHG emissions?"
Response: knowledge

User: "What does the document say about carbon offsets?"
Response: file

User: "Can you analyze the emissions data I uploaded?"
Response: file

User: "What are the three scopes in GHG Protocol?"
Response: knowledge
"""

HIST_SYS = """
You are the History Analyzer for the GHG Protocol chatbot. Your role is to analyze conversation history and respond to queries about past interactions.

INSTRUCTIONS:
- Use ONLY the provided conversation history to answer the user's question
- Summarize previous exchanges accurately and concisely
- When referencing past exchanges, mention when they occurred in the conversation

CONSTRAINTS:
- Never use information outside the provided conversation history
- If the requested information is not in the history, clearly state: "I don't have that information in our conversation history."
- Do not make up answers or fill in missing information
- Keep responses focused specifically on what was discussed
- YOU MUST follow the response length guidelines

TONE:
- Professional and helpful
- Clear and direct
- Objective when summarizing past exchanges

RESPONSE LENGTH:
- Keep responses between 50-150 words
- Use bullet points for summarizing multiple previous exchanges
- Be concise while capturing key information
"""

QDR_SYS = """
You are a GHG Protocol Expert Assistant. Your knowledge comes exclusively from the retrieved document provided in this prompt.

INSTRUCTIONS:
- Answer questions ONLY based on the provided document content

- Begin responses with direct answers before elaborating
- Format technical information clearly using simple language

CONSTRAINTS:
- Never provide information beyond what's in the document
- If the document doesn't contain the answer, state: "Based on the current document, I don't have enough information to answer your question about [topic]."
- Do not make assumptions about GHG Protocol requirements not explicitly stated
- Do not reference information from memory or previous training
- Do not cite specific sections or page numbers when possible
- YOU MUST follow the response length guidelines

TONE:
- Authoritative but accessible
- Technical but clear
- Precise and factual
- Helpful without being overly conversational

RESPONSE LENGTH:
- YOU MUST keep responses under 100 words
- Prioritize short, direct answers with no elaboration unless asked
- Avoid repeating the question or providing background
"""

FILE_SYS = """
You are the File Analyzer for the GHG Protocol chatbot. Your role is to analyze uploaded files and answer questions based solely on their content.

INSTRUCTIONS:
- Answer questions based EXCLUSIVELY on the content of the uploaded file
- When analyzing data, provide specific references to rows, sections, or pages
- For calculations, explain the methodology used from the file
- For documents, summarize relevant sections that answer the query

CONSTRAINTS:
- Never use information outside the provided file content
- If the file doesn't contain relevant information, state: "The uploaded file doesn't contain information about [topic]."
- Do not make assumptions about data not present in the file
- Do not reference external GHG Protocol standards unless cited in the file
- YOU MUST follow the response length guidelines

TONE:
- Analytical and precise
- Factual and objective
- Helpful for data interpretation
- Clear when explaining technical content

RESPONSE LENGTH:
- YOU MUST keep responses under 80 words
- Do not restate the question
- Provide only the answer or insight in bullet or numbered format
"""
KNW_SYS = """
You are the Knowledge Agent for the GHG Protocol chatbot. Your role is to answer common questions about GHG Protocol standards and emissions reporting based on established knowledge.

INSTRUCTIONS:
- Answer common questions about GHG Protocol reporting standards, frequency, scope definitions, and best practices
- Provide factual, accurate information based on the official GHG Protocol guidance
- When referencing specific requirements, mention the relevant protocol or standard

SPECIFIC KNOWLEDGE AREAS:
- GHG Protocol scopes (1, 2, 3) and their definitions
- Standard reporting timeframes and frequency
- Common calculation methodologies
- Verification and assurance requirements
- Industry-specific guidance and sector supplements
- Organizational and operational boundaries
- Common emission factors and their application
- Base year recalculation requirements
- GHG inventories and reporting principles

CONSTRAINTS:
- Stick to established GHG Protocol guidance and standards
- If a question requires very specific document reference, indicate that a document search would be better
- For complex technical questions beyond common knowledge, suggest consulting specific GHG Protocol documents
- Do not make up specific numerical thresholds or deadlines unless they are widely established
- YOU MUST follow the response length guidelines

TONE:
- Authoritative but accessible
- Technical but clear
- Precise and factual
- Helpful and instructive

RESPONSE LENGTH:
- YOU MUST keep responses under 100 words
- Use bullet points for listing requirements or steps
- Keep technical explanations simple and direct
"""


# ---------------------- Utils ----------------------
def _t(txt: str) -> str:
    return txt if len(txt) <= TRUNC else txt[:TRUNC] + "…"

def safe_execute(name, func, *args, **kwargs):
    try:
        logging.info(f"[{name}] Starting")
        res = func(*args, **kwargs)
        logging.info(f"[{name}] Completed")
        return res
    except Exception as e:
        logging.error(f"[{name}] Error: {e}", exc_info=True)
        return None

# ---------------------- Initialize Clients ----------------------
client_aoai = AzureOpenAI(api_key=AZURE_API_KEY, azure_endpoint=AZURE_ENDPOINT, api_version=OPENAI_API_VERSION)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(BLOB_CONTAINER)

# ---------------------- Embedding & Retrieval ----------------------
def get_query_embedding(text: str) -> Optional[List[float]]:
    try:
        resp = client_aoai.embeddings.create(input=[text], model="text-embedding-3-large")
        return resp.data[0].embedding
    except Exception as e:
        logging.error(f"[Embedding] {e}")
        return None

def retrieve_doc_text_and_source(query: str) -> Tuple[Optional[str], Optional[dict], str]:
    """Retrieve relevant document text and source information."""
    vec = get_query_embedding(query)
    if not vec:
        return None, None, "Embedding generation failed"
    
    try:
        res = qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vec,
            limit=1,
            with_payload=True
        )
        
        if not res:
            return None, None, "No relevant document found"
        
        payload = res[0].payload
        fname = payload.get("filename")
        source_link = payload.get("source")
        score = res[0].score
        # Log Qdrant output
        logging.info(f"[Qdrant] found document: {fname} with score: {score}")
        
        blob_name = f"{fname}.docx"
        # Log blob document name
        logging.info(f"[Blob] downloading: {blob_name}")
        
        blob_bytes = container_client.get_blob_client(blob_name).download_blob().readall()
        doc = Document(BytesIO(blob_bytes))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        
        return text, {"documentName": fname, "documentLink": source_link}, ""
    except Exception as e:
        logging.error(f"[Retrieve] error: {e}", exc_info=True)
        return None, None, f"Document retrieval error: {e}"

# ---------------------------------------------------------------------
# Agent runners
# ---------------------------------------------------------------------

def call_model(messages: List[dict], temperature: float = 0) -> str:
    """Call the Azure OpenAI model with provided messages."""
    try:
        resp = client_aoai.chat.completions.create(
            model="gpt-4.1-mini", 
            temperature=temperature, 
            messages=messages
        )
        out = resp.choices[0].message.content.strip()
        return out
    except Exception as e:
        logging.error(f"[ModelCall] error: {e}", exc_info=True)
        return "I'm sorry, I encountered an issue processing your request. Please try again."


def selection_agent(query: str, history_msgs: List[dict] = None, file_content: List[str] = None, mode: str = "simple") -> str:
    """Route the query to the appropriate agent."""
    # Log agent in use
    logging.info(f"[Agent] Selection in mode: {mode}")
    
    if mode == "simple":
        # Simple mode - just route based on the query
        result = call_model([
            {"role": "system", "content": SEL_SYS_SIMPLE},
            {"role": "user", "content": query}
        ]).lower().strip()
    else:
        # Extended mode - consider file history and context
        history_text = ""
        if history_msgs:
            history_text = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}" 
                for msg in history_msgs
            ])
        
        file_text = ""
        if file_content and len(file_content) > 0:
            file_text = "\n".join(file_content)
        
        content = f"""QUERY: {query}

CONVERSATION HISTORY:
{history_text}

PREVIOUSLY UPLOADED FILE CONTENT:
{file_text}

Based on this information, determine if the user is asking about history, general GHG Protocol information, or information related to the uploaded file.
"""
        
        result = call_model([
            {"role": "system", "content": SEL_SYS_EXTENDED},
            {"role": "user", "content": content}
        ]).lower().strip()
    
    # Log agent output
    logging.info(f"[Selection] decided: {result}")
    return result


def history_agent(history_msgs: List[dict], query: str) -> str:
    """Process history-related queries."""
    # Log agent in use
    logging.info(f"[Agent] History")
    
    # Format the conversation history for the prompt
    convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history_msgs)
    
    result = call_model([
        {"role": "system", "content": HIST_SYS},
        {"role": "user", "content": f"CONVERSATION HISTORY:\n{convo}\n\nQUERY: {query}"}
    ])
    
    # Log agent output
    logging.info(f"[History] response length: {len(result)}")
    return result


def qdrant_agent(query: str) -> Tuple[str, Optional[dict]]:
    """Process queries requiring document retrieval."""
    # Log agent in use
    logging.info(f"[Agent] Qdrant")
    
    doc, source, err = retrieve_doc_text_and_source(query)
    if err:
        logging.warning(f"[Qdrant] retrieval error: {err}")
        return f"I couldn't find relevant information about your query. {err}", None
    
    sys_msg = f"{QDR_SYS}\n\nDOCUMENT CONTEXT:\nTitle: {source['documentName']}\nContent:\n{doc}"
    
    ans = call_model([
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": query}
    ])
    
    # Log agent output
    logging.info(f"[Qdrant] response length: {len(ans)}")
    return ans, source


def file_agent(file_text: str, query: str) -> Tuple[str, dict]:
    """Process queries based on uploaded file content."""
    # Log agent in use
    logging.info(f"[Agent] File")
    
    sys_msg = f"{FILE_SYS}\n\nFILE CONTENT:\n{file_text}"
    
    ans = call_model([
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": query}
    ])
    
    # Log agent output
    logging.info(f"[File] response length: {len(ans)}")
    return ans, {"documentName": "Uploaded file", "documentLink": ""}

def knowledge_agent(query: str) -> Tuple[str, dict]:
    """Process common knowledge questions about GHG Protocol standards."""
    # Log agent in use
    logging.info(f"[Agent] Knowledge")
    
    ans = call_model([
        {"role": "system", "content": KNW_SYS},
        {"role": "user", "content": query}
    ])
    
    # Log agent output
    logging.info(f"[Knowledge] response length: {len(ans)}")
    return ans, {"documentName": "GHG Protocol Knowledge Base", "documentLink": "https://ghgprotocol.org/"}

def check_file_history(history_raw):
    contents = []
    for h in history_raw:
        if h.get('user',{}).get('fileUpload') and h['user'].get('fileContent'):
            contents.extend(h['user']['fileContent'])
    return bool(contents), contents

# ---------------------- Main Processing ----------------------
async def process_chatbot_request(data: dict) -> dict:
    d = data.get('data',{})
    history_raw = d.get('history',[])
    history_msgs = []
    for h in history_raw:
        history_msgs.append({'role':'user','content':h['user']['message']})
        history_msgs.append({'role':'assistant','content':h['ai']['response']})

    if d.get('fileUpload'):
        answer, source = file_agent("\n".join(d.get('fileContent',[])), d['latestQuery'])
    else:
        has_files, file_texts = check_file_history(history_raw)
        mode = 'extended' if has_files else 'simple'
        route = selection_agent(d['latestQuery'], history_msgs if has_files else None, file_texts if has_files else None, mode)
        if route=='history': answer, source = history_agent(history_msgs, d['latestQuery']), None
        elif route=='file': answer, source = file_agent("\n".join(file_texts), d['latestQuery'])
        elif route=='knowledge': answer, source = knowledge_agent(d['latestQuery'])
        else: answer, source = qdrant_agent(d['latestQuery'])

    return {'chat':{'response':answer,'sources':source}}

# Session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.history = []

# Initialize saved_sessions mapping from existing JSON files
if 'saved_sessions' not in st.session_state:
    st.session_state.saved_sessions = {}
# Scan session folder for existing session files
for fname in os.listdir(SESSION_DIR):
    if fname.endswith('.json'):
        sid = fname[:-5]
        if sid not in st.session_state.saved_sessions:
            try:
                data = json.load(open(os.path.join(SESSION_DIR, fname)))
                hist = data.get('history', [])
                if hist:
                    first_msg = hist[0]['user']['message']
                    st.session_state.saved_sessions[sid] = first_msg
            except Exception:
                pass

# Sidebar: Saved Sessions
st.sidebar.header("Saved Sessions")
for sid, first_msg in st.session_state.saved_sessions.items():
    if st.sidebar.button(f"{first_msg}", key=sid):
        # load this session
        path = os.path.join(SESSION_DIR, f"{sid}.json")
        if os.path.exists(path):
            data = json.load(open(path))
            st.session_state.session_id = data['sessionId']
            st.session_state.history = data['history']
            st.experimental_rerun()

# Sidebar: New Session button
if st.sidebar.button("New Session"):
    # Save current session
    sid = st.session_state.session_id
    if st.session_state.history:
        first = st.session_state.history[0]['user']['message']
        st.session_state.saved_sessions[sid] = first
        with open(os.path.join(SESSION_DIR, f"{sid}.json"), 'w') as f:
            json.dump({'sessionId': sid, 'history': st.session_state.history}, f, indent=2)
    # Reset state
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.history = []
    st.experimental_rerun()

# Display chat
for msg in st.session_state.history:
    st.chat_message("user").write(msg['user']['message'])
    st.chat_message("assistant").write(msg['ai']['response'])

# Input area
def chat_input_area():
    query = st.chat_input("Your question...")
    return query

query = chat_input_area()
file = st.file_uploader("Upload .docx (optional)", type=['docx'], key='file_uploader')

if query:
    now = datetime.utcnow().isoformat() + 'Z'
    mid = f"msg-{len(st.session_state.history)+1:03d}"
    # file handling
    file_upload = False
    content = []
    if file:
        file_upload = True
        content = [file.read().decode('utf-8', errors='ignore')]
    # build payload
    history_list = [ { 'message_id': h['message_id'], 'createdAt': h['createdAt'], 'user': h['user'], 'ai': h['ai'] } for h in st.session_state.history ]
    payload = {
        'data': {
            'sessionId': st.session_state.session_id,
            'userId': st.session_state.session_id,
            'compId': st.session_state.session_id,
            'latestQuery': query,
            'snippet': '',
            'fileUpload': file_upload,
            'fileContent': content,
            'message_id': mid,
            'createdAt': now,
            'history': history_list
        }
    }
    output = asyncio.run(process_chatbot_request(payload))
    response = output['chat']['response']
    sources = output['chat'].get('sources')
    # append
    entry = { 'message_id': mid, 'createdAt': now,
              'user': {'message': query, 'fileUpload': file_upload, 'fileContent': content, 'read': False},
              'ai': {'response': response, 'sources': sources, 'read': False} }
    st.session_state.history.append(entry)
    # display
    st.chat_message("user").write(query)
    st.chat_message("assistant").write(response)
    # save
    sid = st.session_state.session_id
    with open(os.path.join(SESSION_DIR, f"{sid}.json"), 'w') as f:
        json.dump({'sessionId': sid, 'history': st.session_state.history}, f, indent=2)
