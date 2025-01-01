
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, T5ForConditionalGeneration, T5Tokenizer, Adafactor, get_linear_schedule_with_warmup
import pandas as pd
from sklearn.model_selection import train_test_split

# Load preprocessed data
data = pd.read_csv('Conversation_v2.csv')

# Split the data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=90)

# Load model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('./fine-tuned-model')
model = T5ForConditionalGeneration.from_pretrained('./fine-tuned-model')

# Add padding token if not already present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))  # Adjust model embeddings

# Set a consistent maximum length for padding/truncation
max_length = 512  # Adjust this length as needed

# Tokenize data with consistent padding
def tokenize_data(data):
    questions = data['question'].tolist()
    answers = data['answer'].tolist()

    encodings = tokenizer(questions, 
                          truncation=True, 
                          padding='max_length', 
                          max_length=max_length, 
                          return_tensors='pt',
                          return_attention_mask=True)  # Generate attention masks
    labels = tokenizer(answers, 
                       truncation=True, 
                       padding='max_length', 
                       max_length=max_length, 
                       return_tensors='pt',
                       return_attention_mask=False)

    # Replace padding token id with -100 in labels to ignore them in loss calculation
    labels = labels.input_ids
    labels[labels == tokenizer.pad_token_id] = -100

    return encodings, labels

train_encodings, train_labels = tokenize_data(train_data)
val_encodings, val_labels = tokenize_data(val_data)

# Define dataset class
class LegalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item 

    def __len__(self):
        return len(self.labels)

# Create datasets for training and validation
train_dataset = LegalDataset(train_encodings, train_labels)
val_dataset = LegalDataset(val_encodings, val_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir='./logs',
    logging_steps=100,  # Log every 100 steps
    evaluation_strategy='steps',  # Evaluate at the end of each epoch
    save_steps=500,  # Save checkpoint every 500 steps
    save_total_limit=3,  # Keep only the last 3 checkpoints
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model='eval_loss',  # Use validation loss to choose the best model
    fp16=False,
    gradient_accumulation_steps=3,
    max_grad_norm=1.0
)

# Move model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Define optimizer
optimizer = Adafactor(model.parameters(), lr=1e-4, relative_step=False)
num_training_steps =800
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=68,
    num_training_steps=num_training_steps
)

# Train model with validation
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # Include validation dataset
    optimizers=(optimizer,scheduler) # Pass optimizer (scheduler is None)
)

trainer.train()

# Save final model
model.save_pretrained('./fine-tuned-model')
tokenizer.save_pretrained('./fine-tuned-model')