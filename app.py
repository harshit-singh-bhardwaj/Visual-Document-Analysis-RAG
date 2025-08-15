import streamlit as st
import io
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import camelot
import tempfile
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ----------------- Streamlit Page Config -----------------
st.set_page_config(page_title="Visual Document Analysis RAG", layout="wide")

st.title("📄 Visual Document Analysis RAG System")
st.markdown("""
Upload PDF or image documents (scanned or digital).  
Ask questions on tables, charts, or mixed text-image content.  
This system extracts text with OCR, processes visual info, retrieves relevant chunks, and generates answers.
""")

# ----------------- Document Processing Functions -----------------

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    all_text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if text.strip():
            all_text += text + "\n"
        else:
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            ocr_text = pytesseract.image_to_string(img)
            all_text += ocr_text + "\n"
    pdf_file.seek(0)
    return all_text

def extract_tables_from_pdf(pdf_file):
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(pdf_file.read())
        tmp_pdf_path = tmp_pdf.name

    tables_text = ""
    try:
        tables = camelot.read_pdf(tmp_pdf_path, pages="all")
        for i, table in enumerate(tables):
            tables_text += f"\n[Table {i+1}]\n"
            df = table.df
            headers = df.iloc[0].tolist()
            for row in df.iloc[1:].values:
                row_text = ", ".join([f"{h.strip()}: {v.strip()}" for h, v in zip(headers, row)])
                tables_text += row_text + "\n"
    except Exception as e:
        tables_text += "\n⚠ Could not extract tables: " + str(e)
    pdf_file.seek(0)
    return tables_text

def extract_charts_from_pdf(pdf_file):
    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")
    charts_text = ""
    for page_num in range(pdf.page_count):
        page = pdf.load_page(page_num)
        images = page.get_images(full=True)
        if images:
            for i, img in enumerate(images, 1):
                # Add descriptive placeholder text
                charts_text += f"[Chart {i} on Page {page_num+1} shows some data visually]\n"
    pdf_file.seek(0)
    return charts_text

def extract_text_from_image(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

# ----------------- Improved Chunking -----------------
def chunk_text_and_tables(text, tables_text, chart_text, max_length=150):
    chunks = []

    # Split normal text into chunks
    sentences = text.split("\n")
    current_chunk = ""
    for sent in sentences:
        if len((current_chunk + sent).split()) < max_length:
            current_chunk += sent + " "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sent + " "
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Split table rows into separate chunks
    table_lines = tables_text.split("\n")
    for line in table_lines:
        if line.strip():
            chunks.append("[Table Row] " + line.strip())

    # Add chart placeholders as separate chunks
    if chart_text.strip():
        chart_lines = chart_text.split("\n")
        for line in chart_lines:
            if line.strip():
                chunks.append("[Chart] " + line.strip())

    return chunks

# ----------------- Model & Index Setup -----------------

@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_resource(show_spinner=False)
def load_generator():
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
    return tokenizer, model

def embed_chunks(embedder, chunks):
    embeddings = embedder.encode(chunks, convert_to_tensor=True)
    return embeddings.cpu().detach().numpy()

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_documents(embedder, index, chunks, query, k=5):
    query_emb = embedder.encode([query])
    D, I = index.search(query_emb, k)
    retrieved_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
    return retrieved_chunks

def generate_answer(tokenizer, model, question, retrieved_chunks):
    context = " ".join(retrieved_chunks)
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# ----------------- Streamlit App UI -----------------

uploaded_file = st.file_uploader(
    "Upload PDF or Image (PNG/JPG) Document", 
    type=["pdf", "png", "jpg", "jpeg"]
)

if uploaded_file:
    st.write(f"Uploaded file: {uploaded_file.name} ({uploaded_file.type})")

    with st.spinner("Extracting text and visual content..."):
        if uploaded_file.type == "application/pdf":
            text_content = extract_text_from_pdf(uploaded_file)
            table_text = extract_tables_from_pdf(uploaded_file)
            chart_text = extract_charts_from_pdf(uploaded_file)
            full_text = text_content + "\n" + table_text + "\n" + chart_text
        elif uploaded_file.type in ["image/png", "image/jpeg"]:
            text_content = extract_text_from_image(uploaded_file)
            table_text = ""
            chart_text = ""
            full_text = text_content
        else:
            st.error("Unsupported document format.")
            st.stop()

    st.markdown("### Extracted Document Content Preview")
    st.text_area("Document text:", value=full_text[:5000], height=300)

    chunks = chunk_text_and_tables(text_content, table_text, chart_text)
    st.markdown(f"### Document split into {len(chunks)} chunks for retrieval")

    question = st.text_input("Enter your question about the document:")

    if question and chunks:
        embedder = load_embedder()
        generator_tokenizer, generator_model = load_generator()

        with st.spinner("Encoding chunks and building index..."):
            embeddings = embed_chunks(embedder, chunks)
            index = build_faiss_index(embeddings)

        with st.spinner("Retrieving relevant content..."):
            retrieved_chunks = retrieve_documents(embedder, index, chunks, question, k=5)
            st.markdown("#### Retrieved Context Chunks:")
            for i, chunk in enumerate(retrieved_chunks):
                st.write(f"**Chunk {i+1}:** {chunk[:500]}...")

        with st.spinner("Generating answer..."):
            answer = generate_answer(generator_tokenizer, generator_model, question, retrieved_chunks)

        st.markdown("### Generated Answer")
        st.write(answer)

else:
    st.info("Upload a PDF or image document to start analyzing.")
