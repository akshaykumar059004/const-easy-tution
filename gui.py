import tkinter as tk
from tkinter import scrolledtext
import requests

# Define the URL of the Flask API
API_URL = 'http://localhost:5000/ask'

def get_response():
    """
    Send user input to the Flask API and get the response.
    """
    question = entry.get()
    if not question:
        response_text.config(state=tk.NORMAL)
        response_text.delete('1.0', tk.END)
        response_text.insert(tk.END, "Please enter a question.")
        response_text.config(state=tk.DISABLED)
        return
    
    try:
        response = requests.post(API_URL, json={'question': question})
        response.raise_for_status()
        response_data = response.json()
        answer = response_data.get('response', 'No response received.')
    except requests.exceptions.RequestException as e:
        answer = f"Error: {e}"
    
    response_text.config(state=tk.NORMAL)
    response_text.delete('1.0', tk.END)
    response_text.insert(tk.END, answer)
    response_text.config(state=tk.DISABLED)

# Create the main window
root = tk.Tk()
root.title("Legal Chatbot")

# Create and place widgets
tk.Label(root, text="Enter your question:").pack(pady=5)

entry = tk.Entry(root, width=50)
entry.pack(pady=5)

tk.Button(root, text="Ask", command=get_response).pack(pady=5)

response_text = scrolledtext.ScrolledText(root, height=10, width=50, wrap=tk.WORD, state=tk.DISABLED)
response_text.pack(pady=5)

# Start the GUI event loop
root.mainloop()
