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


@app.route("/upload", methods=["POST"])
def upload_pdf():

    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith(".pdf"):
        pdf_file_path = os.path.join(UPLOAD_FOLDER, file.filename)
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


def query_llama_model(prompt, model="llama3.2:latest"):
    headers = {"Content-Type": "application/json"}

    payload = {"model": model, "prompt": prompt}

    url = "http://localhost:11434/api/generate"

    try:
        response = requests.post(url, json=payload, headers=headers, stream=True)

        if response.status_code == 200:
            full_response = ""
            for chunk in response.iter_lines():
                if chunk:
                    data = json.loads(chunk.decode("utf-8"))
                    full_response += data.get("response", "")

                    if data.get("done", False):
                        break

            return full_response
        else:
            return f"Error: {response.status_code} - {response.text}"

    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"


@app.route("/query", methods=["POST"])
def query_model():
    data = request.get_json()

    if not data or "prompt" not in data:
        return jsonify({"error": "Invalid request, 'prompt' field is required."}), 400

    prompt = data["prompt"]

    response = query_llama_model(prompt)

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
