from flask import Flask, render_template, request, redirect, url_for
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import ollama
from werkzeug.utils import secure_filename


app = Flask(__name__)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# GLOBAL VARIABLES
# ============================================================

pdf_text = ""
chunks = []
faiss_index = None

pdf_name = "No PDF uploaded"

question_count = 0

latest_answer = ""

retrieved_chunk_count = 0

question_history = []


# ============================================================
# UPLOAD FOLDER
# ============================================================

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        pdf_name=pdf_name,
        question_count=question_count,
        latest_answer=latest_answer,
        retrieved_chunk_count=retrieved_chunk_count,
        question_history=question_history
    )


# ============================================================
# UPLOAD PDF
# ============================================================

@app.route("/upload", methods=["POST"])
def upload():

    global pdf_text
    global chunks
    global faiss_index

    global pdf_name
    global question_count
    global latest_answer
    global retrieved_chunk_count
    global question_history


    # --------------------------------------------------------
    # GET PDF
    # --------------------------------------------------------

    pdf = request.files.get("pdf")


    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if not pdf or pdf.filename == "":
        return "Please select a PDF file."


    # --------------------------------------------------------
    # CHECK EXTENSION
    # --------------------------------------------------------

    if not pdf.filename.lower().endswith(".pdf"):
        return "Please upload a PDF file."


    # --------------------------------------------------------
    # SECURE FILENAME
    # --------------------------------------------------------

    filename = secure_filename(pdf.filename)


    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    # --------------------------------------------------------
    # SAVE PDF
    # --------------------------------------------------------

    pdf.save(filepath)


    # Store uploaded PDF name
    pdf_name = filename


    # --------------------------------------------------------
    # RESET DASHBOARD DATA
    # --------------------------------------------------------

    question_count = 0

    latest_answer = ""

    retrieved_chunk_count = 0

    question_history = []


    # ========================================================
    # READ PDF
    # ========================================================

    try:

        reader = PdfReader(filepath)

        pdf_text = ""

        for page in reader.pages:

            extracted = page.extract_text()

            if extracted:

                pdf_text += extracted


    except Exception as e:

        print("\n========== PDF ERROR ==========")

        print(e)

        return "Could not read the uploaded PDF."


    # --------------------------------------------------------
    # CHECK EXTRACTED TEXT
    # --------------------------------------------------------

    if not pdf_text.strip():

        return "Could not extract text from this PDF."


    print("\n========== PDF UPLOADED ==========")

    print("PDF name:", pdf_name)

    print("PDF text length:", len(pdf_text))


    # ========================================================
    # SPLIT PDF INTO CHUNKS
    # ========================================================

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=500,

        chunk_overlap=50

    )


    chunks = splitter.split_text(pdf_text)


    print("Number of chunks:", len(chunks))


    # --------------------------------------------------------
    # CHECK CHUNKS
    # --------------------------------------------------------

    if not chunks:

        return "Could not create text chunks from this PDF."


    # ========================================================
    # CREATE EMBEDDINGS
    # ========================================================

    try:

        embeddings = model.encode(
            chunks
        )

    except Exception as e:

        print("\n========== EMBEDDING ERROR ==========")

        print(e)

        return "Could not create embeddings for this PDF."


    print(
        "Embeddings shape:",
        embeddings.shape
    )


    # ========================================================
    # CREATE FAISS INDEX
    # ========================================================

    try:

        dimension = embeddings.shape[1]

        faiss_index = faiss.IndexFlatL2(
            dimension
        )

        faiss_index.add(
            embeddings
        )

    except Exception as e:

        print("\n========== FAISS ERROR ==========")

        print(e)

        return "Could not create the FAISS index."


    print(
        "Vectors stored in FAISS:",
        faiss_index.ntotal
    )

    print(
        "Document ready for questions."
    )


    return redirect(
        url_for("home")
    )


# ============================================================
# ASK QUESTION
# ============================================================

@app.route("/ask", methods=["POST"])
def ask():

    global question_count
    global latest_answer
    global retrieved_chunk_count
    global question_history


    # --------------------------------------------------------
    # GET QUESTION
    # --------------------------------------------------------

    question = request.form.get(
        "question",
        ""
    ).strip()


    # --------------------------------------------------------
    # CHECK QUESTION
    # --------------------------------------------------------

    if not question:

        return "Please enter a question."


    # --------------------------------------------------------
    # CHECK PDF
    # --------------------------------------------------------

    if faiss_index is None:

        return "Please upload a PDF first."


    # ========================================================
    # QUESTION COUNT + HISTORY
    # ========================================================

    question_count += 1

    question_history.append(
        question
    )


    print("\n========== QUESTION ==========")

    print(question)


    # ========================================================
    # CREATE QUESTION EMBEDDING
    # ========================================================

    try:

        question_embedding = model.encode(
            [question]
        )

    except Exception as e:

        print(
            "\n========== QUESTION EMBEDDING ERROR =========="
        )

        print(e)

        return "Could not process the question."


    # ========================================================
    # FAISS SEARCH
    # ========================================================

    try:

        distances, indices = faiss_index.search(

            question_embedding,

            10

        )

    except Exception as e:

        print(
            "\n========== FAISS SEARCH ERROR =========="
        )

        print(e)

        return "Could not search the uploaded document."


    # ========================================================
    # GET RELEVANT CHUNKS + NEIGHBOURING CHUNKS
    # ========================================================

    relevant_chunks = []


    for index in indices[0]:

        # Ignore invalid FAISS index
        if index < 0 or index >= len(chunks):

            continue


        start = max(
            0,
            index - 1
        )


        end = min(
            len(chunks),
            index + 2
        )


        for i in range(
            start,
            end
        ):

            if chunks[i] not in relevant_chunks:

                relevant_chunks.append(
                    chunks[i]
                )


    # ========================================================
    # STORE RETRIEVED CHUNK COUNT
    # ========================================================

    retrieved_chunk_count = len(
        relevant_chunks
    )


    # ========================================================
    # COMBINE CONTEXT
    # ========================================================

    context = "\n\n".join(
        relevant_chunks
    )


    # ========================================================
    # PRINT RETRIEVED CONTEXT
    # ========================================================

    print(
        "\n========== RELEVANT CONTEXT RETRIEVED =========="
    )


    for i, chunk in enumerate(
        relevant_chunks
    ):

        print(
            f"\n--- Relevant Chunk {i + 1} ---"
        )

        print(chunk)


    print(
        "\nTotal context chunks:",
        retrieved_chunk_count
    )


    # ========================================================
    # PROMPT FOR LLAMA
    # ========================================================

    prompt = f"""
You are FileChat, a document question-answering assistant.

Your task is to answer the user's question using ONLY
the information contained in the PDF context provided below.

IMPORTANT RULES:

1. Use only information from the PDF context.

2. Do not use outside knowledge.

3. Do not invent or assume information.

4. Carefully examine ALL provided context before answering.

5. If the answer is spread across multiple chunks,
   combine the relevant information.

6. For questions asking for numbers, counts, units,
   modules, chapters, topics, lists, or totals, carefully
   examine all relevant context and give the complete answer.

7. If the PDF clearly states the answer, answer directly.

8. If the answer cannot be found in the provided context,
   respond exactly with:

"I could not find the answer in the uploaded PDF."

9. Keep the answer clear and concise.

PDF CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""


    # ========================================================
    # SEND QUESTION TO LLAMA
    # ========================================================

    print(
        "\n========== SENDING TO LLAMA =========="
    )


    try:

        response = ollama.chat(

            model="llama3.2:3b",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]

        )


    except Exception as e:

        print(
            "\n========== OLLAMA ERROR =========="
        )

        print(e)


        return """
        <h3>AI service unavailable</h3>

        <p>
        Please make sure Ollama is running and the
        Llama 3.2 model is installed.
        </p>
        """


    # ========================================================
    # GET LLAMA ANSWER
    # ========================================================

    try:

        answer = response[
            "message"
        ][
            "content"
        ]

    except Exception as e:

        print(
            "\n========== LLAMA RESPONSE ERROR =========="
        )

        print(e)

        return "Could not process the AI response."


    # ========================================================
    # STORE LATEST ANSWER
    # ========================================================

    latest_answer = answer


    print(
        "\n========== LLAMA ANSWER =========="
    )

    print(answer)


    # ========================================================
    # RETURN DASHBOARD
    # ========================================================

    return render_template(

        "index.html",

        pdf_name=pdf_name,

        question_count=question_count,

        latest_answer=latest_answer,

        retrieved_chunk_count=retrieved_chunk_count,

        question_history=question_history

    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
