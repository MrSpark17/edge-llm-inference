from fastapi import FastAPI
import ollama

app = FastAPI()

@app.get("/")
def home():
	return {"message": "Welcome to the DeepScaleR Chatbot API"}

@app.get("/chat")
def chat(message: str):
	response = ollama.chat(model='deepscaler-chat', messages=[{"role": "user", "content": message}])
	return {"response": response["message"]}

