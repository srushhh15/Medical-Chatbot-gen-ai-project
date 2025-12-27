from flask import Flask, request, render_template, session, redirect, url_for
import os

# ====== YOUR RAG IMPORTS (already working in your notebook) ======
from groq import Groq
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

app = Flask(__name__)
app.secret_key = "super-secret-key"  # required for sessions

# ====== INITIALIZE SERVICES ======
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("medical-chatbot-index")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ====== RAG FUNCTIONS ======
def retrieve_docs(query, top_k=5):
    results = index.query(
        vector=embeddings.embed_query(query),
        top_k=top_k,
        include_metadata=True
    )
    return results["matches"]


def build_context(matches):
    contexts = []
    for match in matches:
        if "text" in match["metadata"]:
            contexts.append(match["metadata"]["text"])
    return "\n\n".join(contexts)


def groq_medical_answer(context, question):
    prompt = f"""
You are a medical assistant.
Answer ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_completion_tokens=512
    )

    return response.choices[0].message.content


def get_medical_answer(query):
    matches = retrieve_docs(query)
    if not matches:
        return "I don't know based on the available medical documents."
    context = build_context(matches)
    return groq_medical_answer(context, query)


# ====== ROUTES ======
@app.route("/", methods=["GET", "POST"])
def home():
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        query = request.form.get("question")
        if query:
            answer = get_medical_answer(query)

            session["chat_history"].append({
                "question": query,
                "answer": answer
            })
            session.modified = True

    return render_template("index.html", chat_history=session["chat_history"])


@app.route("/reset", methods=["POST"])
def reset():
    session.pop("chat_history", None)
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

