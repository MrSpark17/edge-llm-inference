import gradio as gr
import ollama

# Function to interact with the Deepscaler chatbot
def chat_with_bot(user_message):
	response = ollama.chat(model='deepscaler-chat', messages=[{"role": "user", "content": user_message}])
	return response["message"]

# Define Geadio Chatbot Interface
chatbot_ui = gr.Interface(
	fn = chat_with_bot,
	inputs=gr.Textbox(label="Chat Message"),
	outputs=gr.Textbox(label="Bot Response"),
	title="AI-Chatbot using DeepScaler",
	description="Chat with an AI chatbot powered by DeepScaler and Ollama."
)

if __name__ == "__main__":
	chatbot_ui.launch()
