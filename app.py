from flask import Flask, render_template, request
from dotenv import load_dotenv
from pathlib import Path
import os
import torch

from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore

from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================================================
# Flask
# =========================================================

app = Flask(__name__)

load_dotenv()


# =========================================================
# Environment
# =========================================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY is missing from your .env file."
    )

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


# =========================================================
# Device
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Medical Chatbot")
print("=" * 60)
print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")


# =========================================================
# Hugging Face Local LLM
# =========================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("\nLoading Hugging Face model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto"
)

model = model.to(device)
model.eval()

print("Hugging Face model loaded successfully.")


# =========================================================
# Hugging Face Embeddings
# =========================================================

print("\nLoading embedding model...")

embeddings = download_hugging_face_embeddings()

print("Embedding model loaded.")


# =========================================================
# Pinecone
# =========================================================

index_name = "medical-chatbot"

print("\nConnecting to Pinecone...")

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3
    }
)

print("Pinecone connected successfully.")


# =========================================================
# Local Hugging Face Answer Generator
# =========================================================

def generate_answer(question, context):

    prompt = f"""
You are MedCare AI, a medical information assistant.

Your job is to answer the user's question using the medical
information retrieved from the provided medical book.

IMPORTANT RULES:

1. Use the provided context as your primary source.
2. Do not invent facts.
3. Do not claim to be a real doctor.
4. Do not diagnose the user.
5. Do not prescribe medicines or dosages.
6. If the context does not contain enough information,
   clearly say that the information is not available
   in the provided medical book.
7. Give a clear, helpful and easy-to-understand answer.
8. For emergency symptoms, advise the user to seek
   immediate professional medical care.

MEDICAL BOOK CONTEXT:

{context}

USER QUESTION:

{question}

Provide a clear answer:

"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False
        )

    input_length = inputs["input_ids"].shape[1]

    answer = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    )

    return answer.strip()


# =========================================================
# RAG Function
# =========================================================

def ask_medical_question(question):

    docs = retriever.invoke(question)

    if not docs:
        return (
            "I could not find relevant information in "
            "the medical book."
        )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    answer = generate_answer(
        question,
        context
    )

    return answer


# =========================================================
# PAGE 1 — HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# PAGE 2 — CHATBOT
# =========================================================

@app.route("/chat")
def chatbot():

    return render_template(
        "chat.html"
    )


# =========================================================
# CHAT API
# =========================================================

@app.route("/get", methods=["POST"])
def chat():

    try:

        msg = request.form.get(
            "msg",
            ""
        ).strip()

        if not msg:

            return "Please enter a medical question."

        print("\nUser:", msg)

        response = ask_medical_question(
            msg
        )

        print("AI:", response)

        return response

    except Exception as e:

        print("\nERROR:")
        print(type(e).__name__)
        print(e)

        return (
            "Sorry, I couldn't process your question "
            "right now. Please try again."
        )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8080,
        debug=False,
        use_reloader=False
    )