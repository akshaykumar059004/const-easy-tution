from flask import Flask, request, jsonify
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)

# Load model and tokenizer
model_dir = './fine-tuned-model'
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)

# Ensure the model is in evaluation mode
model.eval()

def generate_response(input_text):
    """
    Generate a response from the model for a given input text.
    """
    # Tokenize the input text
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    
    # Generate a response from the model
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the generated tokens
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.route('/ask', methods=['POST'])
def ask():
    """
    Handle POST requests to the /ask endpoint.
    Expects a JSON payload with a 'question' field.
    """
    # Get JSON data from the request
    data = request.get_json()
    
    # Check if 'question' field is in the request
    if 'question' not in data:
        return jsonify({'error': 'Missing question field'}), 400
    
    question = data['question']
    
    # Generate and return the response
    response = generate_response(question)
    return jsonify({'response': response})

if __name__ == "__main__":
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000)
