from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model & vectorizer
model = pickle.load(open("model/svm_sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("model/atfidf_vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    tweet = ""

    if request.method == "POST":
        tweet = request.form["tweet"]
        tweet_vector = vectorizer.transform([tweet])
        pred = model.predict(tweet_vector)[0]

        # Adjust labels based on your model
        if pred == 1:
            prediction = "Positive 😊"
        elif pred == 0:
            prediction = "Negative 😠"
        else:
            prediction = "Neutral 😐"

    return render_template("index.html", prediction=prediction, tweet=tweet)

if __name__ == "__main__":
    app.run(debug=True)
