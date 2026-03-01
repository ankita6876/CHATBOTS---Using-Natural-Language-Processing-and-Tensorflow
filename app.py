from flask import Flask, render_template, request, jsonify
import numpy as np
import json
import random
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model
model = load_model("chatbot_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load intents
with open("intents.json") as file:
    data = json.load(file)

# Prepare labels
labels = sorted([intent["tag"] for intent in data["intents"]])

context = None

def get_response(user_input):
    global context
    
    seq = tokenizer.texts_to_sequences([user_input])
    padded_seq = pad_sequences(seq, maxlen=model.input_shape[1])
    
    result = model.predict(padded_seq)
    index = np.argmax(result)
    confidence = result[0][index]
    
    if confidence > 0.6:
        tag = labels[index]
        
        for intent in data["intents"]:
            if intent["tag"] == tag:
                
                if "context_filter" in intent:
                    if context != intent["context_filter"]:
                        continue
                
                if "context_set" in intent:
                    context = intent["context_set"]
                
                return random.choice(intent["responses"])
    
    return "Sorry, I don't understand."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_input = request.form["message"]
    response = get_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)