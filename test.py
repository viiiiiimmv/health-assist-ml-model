from flask import Flask, request, jsonify

app = Flask(__name__)

# Example ML logic (replace with your actual model)
def predict_response(user_input):
    if "fever" in user_input.lower():
        return "It sounds like you may have a fever. Please stay hydrated and rest."
    elif "headache" in user_input.lower():
        return "Headaches can have many causes. Try drinking water and resting."
    else:
        return "I'm here to help! Could you describe your symptoms in more detail?"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user_input = data.get("input", "")
    response = predict_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(port=5173, debug=True)
