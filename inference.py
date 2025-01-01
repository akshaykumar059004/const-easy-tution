import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, T5ForConditionalGeneration, T5Tokenizer

def load_model_and_tokenizer(model_dir):
    # Load the fine-tuned model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    return tokenizer, model

def generate_response(model, tokenizer, input_text):
    try:
        # Tokenize the input text with attention masks
        inputs = tokenizer.encode_plus(
            input_text, 
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=512,  # Adjust max_length based on your model's training
            return_attention_mask=True  # Include attention mask
        )
        
        input_ids = inputs['input_ids'].to(model.device)
        attention_mask = inputs['attention_mask'].to(model.device)

        # Generate a response from the model
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,  # Pass the attention mask
                max_new_tokens=150,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,  # EOS token for padding
                repetition_penalty=1.2,
                temperature=0.7,  # Adjust for your needs; higher for more randomness
                top_k=50,         # Limits the sampling pool to top 50 tokens
                top_p=0.95,       # Nucleus sampling; adjust to control diversity
                num_beams=5,
                do_sample=True,
            )
        
        # Decode the generated tokens
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

def main():
    # Directory where the fine-tuned model is saved
    model_dir = './fine-tuned-model'
    
    # Load the model and tokenizer
    tokenizer, model = load_model_and_tokenizer(model_dir)
    
    # Move model to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Put the model in evaluation mode
    model.eval()
    
    print("Enter 'exit' to end the session.")
    
    while True:
        # Get user input
        input_text = input("You: ")
        
        if input_text.lower() == 'exit':
            break
        
        # Token length notification
        if len(tokenizer.tokenize(input_text)) > 512:
            print("Warning: Your input is too long and will be truncated.")
        
        # Generate and print the response
        response = generate_response(model, tokenizer, input_text)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
