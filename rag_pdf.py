import json
import os

import PyPDF2
import requests
import torch
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

UPLOAD_FOLDER = "./uploaded_pdfs"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")

pdf_texts = {}


def extract_text_from_pdf(pdf_file_path):
    with open(pdf_file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        extracted_text = ""
        for page in range(len(reader.pages)):
            extracted_text += reader.pages[page].extract_text()
        return extracted_text


def retrieve_relevant_text(query, sentences, embeddings, top_k=5):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, embeddings)[0]

    top_results = torch.topk(similarities, k=top_k)

    relevant_text = "\n".join([sentences[idx] for idx in top_results.indices])
    return relevant_text


def query_llama_model(prompt, retrieved_text, model="llama3.2:latest"):
    headers = {"Content-Type": "application/json"}

    full_prompt = f"Context: {retrieved_text}\n\nQuestion: {prompt}\nAnswer:"
    payload = {"model": model, "prompt": full_prompt}

    url = "http://localhost:11434/api/generate"
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        return response.json().get("response", "No response")
    else:
        return f"Error: {response.status_code} - {response.text}"


@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith(".pdf"):
        pdf_file_path = os.path.join("./uploaded_pdfs", file.filename)
        file.save(pdf_file_path)

        extracted_text = extract_text_from_pdf(pdf_file_path)

        sentences = extracted_text.split("\n")
        embeddings = embedder.encode(sentences, convert_to_tensor=True)

        pdf_id = len(pdf_texts) + 1
        pdf_texts[pdf_id] = {
            "text": sentences,
            "embeddings": embeddings,
            "file_path": pdf_file_path,
        }

        return jsonify({"message": "PDF uploaded successfully", "pdf_id": pdf_id})
    else:
        return jsonify({"error": "Only PDF files are allowed."}), 400


@app.route("/query", methods=["POST"])
def query_pdf():
    data = request.get_json()

    if not data or "prompt" not in data or "pdf_id" not in data:
        return (
            jsonify({"error": "Invalid request, 'prompt' and 'pdf_id' are required"}),
            400,
        )

    prompt = data["prompt"]
    pdf_id = data["pdf_id"]
    print("pdf_texts", pdf_texts)

    if pdf_id not in pdf_texts:
        return jsonify({"error": "Invalid PDF ID"}), 400

    pdf_data = pdf_texts[pdf_id]
    embeddings = pdf_data["embeddings"]
    sentences = pdf_data["text"]

    relevant_text = retrieve_relevant_text(prompt, sentences, embeddings)

    response = query_llama_model(prompt, relevant_text)

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
