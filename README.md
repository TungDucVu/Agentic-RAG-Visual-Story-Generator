# 📖 Agentic RAG Visual Story Generator

An **Agentic AI-powered visual storytelling system** that generates rich, creative narratives from images using **Multimodal LLMs + Retrieval-Augmented Generation (RAG)**.

---

## 🚀 Overview

This project combines:

- 🧠 **LLM (GPT-4o)** for story generation  
- 🖼️ **Multimodal input (images + text)**  
- 🔎 **RAG (Retrieval-Augmented Generation)** for contextual storytelling  
- 🧩 **LangGraph Agentic Workflow** for structured reasoning  
- 🗃️ **Chroma Vector DB** for similarity search  
- ☁️ **Cloudinary** for image hosting  
- 🎙️ **Text-to-Speech** for narration  
- 🌐 **Streamlit UI** for interaction  

👉 The system takes an uploaded image and generates a **context-aware story** by retrieving similar image-text pairs and blending them creatively.

---

## 🧠 How It Works

### 🔁 Pipeline Flow

1. Image Upload  
2. Image Captioning (LLM Tool Node)  
3. Dynamic Query Generation  
4. Vector Search (MMR Retrieval)  
5. Context Extraction (Text + Images)  
6. Multimodal Story Generation (LLM)  
7. Output + Voice Narration  

---

## 🏗️ Architecture

```
User Image
   ↓
LLM (Caption Generator)
   ↓
Dynamic Query
   ↓
Chroma DB (MMR Search)
   ↓
Retrieved Stories + Images
   ↓
Multimodal Prompt
   ↓
GPT-4o
   ↓
Generated Story + Audio
```

---

## ⚙️ Tech Stack

| Component        | Technology |
|----------------|----------|
| LLM            | GPT-4o (via OpenAI-compatible API) |
| Embeddings     | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB      | Chroma |
| Workflow       | LangGraph |
| Backend        | Python |
| UI             | Streamlit |
| Image Hosting  | Cloudinary |
| TTS            | pyttsx3 |

---

## 📂 Project Structure

```
├── Agentic_Workflow.py
├── chroma_store/
├── .env
├── requirements.txt
└── README.md
```

---

## 🔑 Environment Variables

Create a `.env` file:

```
LLM_API_KEY=your_api_key
LLM_BASE_URL=your_base_url

CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
```

---

## 📦 Installation

```bash
git clone https://github.com/your-username/agentic-rag-story-generator.git
cd agentic-rag-story-generator
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
streamlit run Agentic_Workflow.py
```

---

## 🧩 Key Features

- Retrieval-Augmented Generation (RAG)  
- Agentic Workflow (LangGraph)  
- Multimodal Reasoning (Image + Text)  
- Max Marginal Relevance (MMR) Search  
- Text-to-Speech Output  

---

## ⚠️ Limitations

- Requires external APIs  
- Performance depends on dataset quality  
- Base64 conversion may slow down processing  

---

## 🔮 Future Improvements (not yet done)

- Hybrid search (image + text embeddings)  
- Better caching & optimization  
- Multi-step storytelling memory  
- Full deployment (FastAPI + frontend)  

---

## 📜 License

MIT License
