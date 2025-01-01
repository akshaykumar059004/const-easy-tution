import tkinter as tk
from tkinter import font
import requests

# Define the URL of the Flask API
API_URL = 'http://localhost:5000/ask'

# Placeholder flag
first_message_sent = False

def get_response():
    global first_message_sent
    
    # Remove the placeholder text when the first message is sent
    if not first_message_sent:
        response_text.config(state=tk.NORMAL)
        response_text.delete("1.0", tk.END)
        response_text.config(state=tk.DISABLED)
        first_message_sent = True

    question = entry.get()
    if not question:
        return

    try:
        response = requests.post(API_URL, json={'question': question})
        response.raise_for_status()
        response_data = response.json()
        answer = response_data.get('response', 'No response received.')
    except requests.exceptions.RequestException as e:
        answer = f"Error: {e}"

    display_message("You", question, is_user=True)
    display_message("Const-Easy-Tution", answer)

    entry.delete(0, tk.END)

def display_message(sender, message, is_user=False):
    """
    Display a message in the chat window with bubble-like appearance.
    """
    response_text.config(state=tk.NORMAL)
    
    tag_name = "user_msg" if is_user else "bot_msg"
    
    response_text.insert(tk.END, f"{sender}:\n", "sender")
    response_text.insert(tk.END, f"{message}\n\n", tag_name)
    
    response_text.config(state=tk.DISABLED)
    response_text.see(tk.END)

# Create the main window
root = tk.Tk()
root.title("Const-Easy-Tution")

# Set window size and make it unresizable
root.geometry("900x700")
root.resizable(False, False)

# Dark theme colors
bg_color = "#1e1e1e"
text_color = "#e1e1e1"
user_bubble_color = "#4a90e2"
bot_bubble_color = "#333333"
button_color = "#0084FF"
entry_bg_color = "#2b2b2b"

# Fonts
sender_font = font.Font(family="Helvetica Neue", size=12, weight="bold")
message_font = font.Font(family="Helvetica Neue", size=12)

# Frame for chat display
frame = tk.Frame(root, bg=bg_color)
frame.pack(fill=tk.BOTH, expand=True)

# Custom Scrollbar
scrollbar = tk.Scrollbar(frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Chat display area with custom tags for message styling
response_text = tk.Text(frame, height=25, width=75, wrap=tk.WORD, bg=bg_color, fg=text_color, font=message_font, yscrollcommand=scrollbar.set, state=tk.DISABLED, bd=0, padx=10, pady=10, relief=tk.FLAT)
response_text.tag_configure("sender", font=sender_font, foreground="#FFFFFF")

# Flat appearance for text bubbles
response_text.tag_configure("user_msg", 
    background=user_bubble_color, 
    foreground="#FFFFFF", 
    lmargin1=20, 
    lmargin2=20, 
    rmargin=20, 
    spacing1=10,  # Increased space above the bubble
    spacing2=5,   # Space below the bubble, reduced to create gap
    spacing3=5, 
    borderwidth=0, 
    relief=tk.FLAT)

response_text.tag_configure("bot_msg", 
    background=bot_bubble_color, 
    foreground=text_color, 
    lmargin1=20, 
    lmargin2=20, 
    rmargin=20, 
    spacing1=5,   # Space above the bubble
    spacing2=10,  # Increased space below the bubble, creating gap
    spacing3=5, 
    borderwidth=0, 
    relief=tk.FLAT)

response_text.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
scrollbar.config(command=response_text.yview)

# Display the centered placeholder text initially
response_text.config(state=tk.NORMAL)
response_text.insert(tk.END, "\n\n\n\n\n\nConst-Easy-Tution - An AI powered legal chatbot", "centered")
response_text.tag_configure("centered", justify='center', font=("Helvetica Neue", 18, "bold"), foreground=text_color)
response_text.config(state=tk.DISABLED)

# Frame for entry and send button
entry_frame = tk.Frame(frame, bg=bg_color)
entry_frame.pack(fill=tk.X, padx=20, pady=10)

# Entry widget with rounded corners and placeholder text
entry = tk.Entry(entry_frame, font=message_font, bg=entry_bg_color, fg=text_color, insertbackground=text_color, bd=0, highlightthickness=0, relief=tk.FLAT)
entry.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.X, expand=True)
entry.insert(0, "Type your message here...")

# Function to clear placeholder text on click
def on_entry_click(event):
    if entry.get() == "Type your message here...":
        entry.delete(0, tk.END)
        entry.config(fg=text_color)

# Function to restore placeholder text if entry is empty
def on_focus_out(event):
    if entry.get() == "":
        entry.insert(0, "Type your message here...")
        entry.config(fg=text_color)

entry.bind('<FocusIn>', on_entry_click)
entry.bind('<FocusOut>', on_focus_out)

# Bind the Enter key to the get_response function
entry.bind('<Return>', lambda event: get_response())

# Send button with modern styling and hover effect
send_button = tk.Button(entry_frame, text="Send", command=get_response, bg=button_color, fg="white", font=message_font, relief=tk.FLAT, bd=0, activebackground="#005bb5", activeforeground="white")
send_button.pack(side=tk.RIGHT, padx=10, pady=10)

# Hover effect for the send button
def on_enter(e):
    send_button.config(bg="#005bb5")

def on_leave(e):
    send_button.config(bg=button_color)

send_button.bind("<Enter>", on_enter)
send_button.bind("<Leave>", on_leave)

# Start the GUI event loop
root.mainloop()
