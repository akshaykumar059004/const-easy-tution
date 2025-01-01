import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_metric
import pandas as pd
from tqdm import tqdm
import os

# Load the fine-tuned model and tokenizer
model_dir = './fine-tuned-model'
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)

# Move model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Load the evaluation dataset
data = pd.read_csv('preprocessed_legal_data.csv')  # Assuming validation data is saved as a CSV file
questions = data['question'].tolist()
answers = data['answer'].tolist()

# Initialize evaluation metrics
bleu_metric = load_metric("bleu", trust_remote_code=True)
rouge_metric = load_metric("rouge")

def evaluate_model(model, tokenizer, questions, answers):
    predictions = []
    
    for question in tqdm(questions):
        # Tokenize the input text with attention mask
        inputs = tokenizer(question, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Generate a response from the model
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,  # Pass the attention mask to the model
                max_length=150, 
                num_return_sequences=1, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,          # Enable sampling
                top_k=50,                # Top-k sampling
                top_p=0.95,              # Top-p (nucleus) sampling
                temperature=0.7,         # Temperature for diversity
                repetition_penalty=1.2   # Penalty for repetition
            )
        
        # Decode the generated tokens
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(response)
    
    # Calculate BLEU score
    references = [[answer.split()] for answer in answers]
    predicted_tokens = [pred.split() for pred in predictions]
    bleu_score = bleu_metric.compute(predictions=predicted_tokens, references=references)
    
    # Calculate ROUGE score
    rouge_score = rouge_metric.compute(predictions=predictions, references=answers)
    
    return bleu_score, rouge_score, predictions

# Evaluate the model
bleu_score, rouge_score, predictions = evaluate_model(model, tokenizer, questions, answers)

# Print evaluation results
print(f"BLEU Score: {bleu_score['bleu']}")
print(f"ROUGE Score: {rouge_score}")

# Save the evaluation results to a CSV file
evaluation_results = {
    'BLEU Score': [bleu_score['bleu']],
    'ROUGE-1 Precision': [rouge_score['rouge1'].mid.precision],
    'ROUGE-1 Recall': [rouge_score['rouge1'].mid.recall],
    'ROUGE-1 F1': [rouge_score['rouge1'].mid.fmeasure],
    'ROUGE-2 Precision': [rouge_score['rouge2'].mid.precision],
    'ROUGE-2 Recall': [rouge_score['rouge2'].mid.recall],
    'ROUGE-2 F1': [rouge_score['rouge2'].mid.fmeasure],
    'ROUGE-L Precision': [rouge_score['rougeL'].mid.precision],
    'ROUGE-L Recall': [rouge_score['rougeL'].mid.recall],
    'ROUGE-L F1': [rouge_score['rougeL'].mid.fmeasure],
    'ROUGE-Lsum Precision': [rouge_score['rougeLsum'].mid.precision],
    'ROUGE-Lsum Recall': [rouge_score['rougeLsum'].mid.recall],
    'ROUGE-Lsum F1': [rouge_score['rougeLsum'].mid.fmeasure],
}

results_df = pd.DataFrame(evaluation_results)
results_file = 'evaluation_results_t5.csv'
if os.path.exists(results_file):
    results_df.to_csv(results_file, mode='a', header=False, index=False)
else:
    results_df.to_csv(results_file, index=False)

# Optionally, save the predictions to a file for further analysis
data['Predictions'] = predictions
data.to_csv('model_predictions.csv', index=False)

# Clear GPU memory
torch.cuda.empty_cache()
