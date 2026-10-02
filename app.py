from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# Load pre-trained BART summarization model
print("Loading AI model... Please wait.")

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/bart-large-cnn"
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/bart-large-cnn"
)

print("AI model loaded successfully!")


# Home page
@app.route("/")
def home():
    return send_from_directory(".", "index.html")


# Summary API
@app.route("/summarize", methods=["POST"])
def summarize():

    try:

        # Get data from website
        data = request.get_json()

        text = data.get("text", "").strip()

        # Check input
        if not text:
            return jsonify({
                "error": "Please enter some text."
            }), 400

        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )

        # Generate summary
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=80,
            min_length=20,
            do_sample=False
        )

        # Convert tokens to text
        summary = tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        # Word counts
        original_words = len(text.split())
        summary_words = len(summary.split())

        # Percentage reduction
        reduction = (
            (original_words - summary_words)
            / original_words
        ) * 100

        # Send results to website
        return jsonify({
            "summary": summary,
            "original_words": original_words,
            "summary_words": summary_words,
            "reduction": reduction
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# Start Flask server
if __name__ == "__main__":
    app.run(debug=True)