from flask import Flask, render_template, request, redirect, url_for, session
import os

# --------------------------------------------------
# Flask setup
# --------------------------------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey"  # required for session

# --------------------------------------------------
# Dummy answer function
# Replace this with your Groq + Pinecone RAG logic
# --------------------------------------------------
def get_medical_answer(query):
    # 🔁 Replace this with:
    # matches = retrieve_docs(query)
    # context = build_context(matches)
    # answer = groq_medical_answer(context, query)
    return "Diabetes mellitus — A metabolic disease caused by a deficiency of insulin, which is essential to process carbohydrates in the body."

# --------------------------------------------------
# Home route
# --------------------------------------------------
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

        return redirect(url_for("home"))

    return render_template("index.html", chat_history=session["chat_history"])

# --------------------------------------------------
# Reset chat route (NO 405 error)
# --------------------------------------------------
@app.route("/reset", methods=["GET", "POST"])
def reset():
    session["chat_history"] = []
    session.modified = True
    return redirect(url_for("home"))

# --------------------------------------------------
# Run app
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
