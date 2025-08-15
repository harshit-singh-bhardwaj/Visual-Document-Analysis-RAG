# Visual Document Analysis RAG

An AI-powered application for analyzing and querying visual documents (PDFs, scanned images, etc.) using a Retrieval-Augmented Generation (RAG) pipeline.  
Built with **Streamlit**, **LangChain**, and **OCR** tools to extract text and provide intelligent answers to user queries.

---

## 🚀 Features
- Upload PDFs and image-based documents.
- Extract text from scanned pages using OCR.
- Store and retrieve document embeddings for fast query answering.
- Context-aware Q&A using LLMs.
- User-friendly interface powered by Streamlit.

---

## 🛠️ Tech Stack
- **Python 3.12**
- **Streamlit** – Web app UI
- **LangChain** – RAG pipeline
- **FAISS** – Vector database for embeddings
- **Tesseract OCR** – Extract text from images
- **OpenAI / LLaMA / HuggingFace models** – LLM integration

---

## 📦 Installation
1. **Clone the repository**
```bash
git clone https://github.com/harshit-singh-bhardwaj/Visual-Document-Analysis-RAG.git
cd Visual-Document-Analysis-RAG

Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

Install dependencies
pip install -r requirements.txt

Set environment variables
Create a .env file in the root directory and add:
OPENAI_API_KEY=your_api_key_here

Run the app locally
streamlit run app.py

🌐 Deployment
The app is deployed using Streamlit Cloud.
You can try it here: Live App Link

📄 Project Structure
.
├── app.py                # Main Streamlit application
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
├── data/                 # Sample documents
└── .env.example          # Environment variable template

📜 License
This project is licensed under the MIT License.

✨ Author
Harshit Singh Bhardwaj
