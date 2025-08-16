# 📄 Visual Document Analysis with RAG  

**An AI-powered application for intelligent document understanding and Q&A**  
Built with **LangChain, OpenAI, and PyMuPDF**, this project uses **Retrieval-Augmented Generation (RAG)** to answer natural language queries from uploaded PDFs.  

---

## 🚀 Features

- Upload PDFs and image-based documents.
- Extract text from scanned pages using OCR.
- Store and retrieve document embeddings for fast query answering.
- Context-aware Q&A using LLMs.
- User-friendly interface powered by Streamlit.

---

## 🛠 Tech Stack

- **Python 3.12**
- **Streamlit** – Web app UI
- **LangChain** – RAG pipeline
- **FAISS** – Vector database for embeddings
- **Tesseract OCR** – Extract text from images
- **OpenAI / LLaMA / HuggingFace models** – LLM integration

---

## 📂 Project Structure

```plaintext
Visual-Document-Analysis-RAG/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── LICENSE                   # MIT License
```
---

## 📦 Installation  

**1️⃣ Clone the repository**  
git clone https://github.com/harshit-singh-bhardwaj/Visual-Document-Analysis-RAG.git  
cd Visual-Document-Analysis-RAG

**2️⃣ Create virtual environment**  
```plaintext
python -m venv venv  
source venv/bin/activate         # Mac/Linux  
venv\Scripts\activate            # Windows
```

**3️⃣ Install dependencies**  
pip install -r requirements.txt

**4️⃣ Set environment variable for OpenAI API key** 
```plaintext
export OPENAI_API_KEY="your_api_key"   # Mac/Linux  
setx OPENAI_API_KEY "your_api_key"     # Windows
```

---
## 📸 Screenshots
<img width="1403" height="488" alt="Screenshot 2025-08-16 at 2 06 59 AM" src="https://github.com/user-attachments/assets/eba78d79-10b2-49d1-9618-054a7fb13eab" />

---

## ▶️ Usage
streamlit run app.py
1. Upload a PDF document.
2. Enter your query in the text box.
3. View AI-generated responses based on your document content.

---

## 🌐 Deployment
The app is deployed using Streamlit Cloud.  
You can try it here: https://visual-document-analysis-rag.streamlit.app

---

## 📜 License
This project is licensed under the MIT License.

---

## ✨ Author
Harshit Singh Bhardwaj
