from flask import Flask, request, jsonify
from flask_cors import CORS  # Importez CORS
from rag import RAG  # Assurez-vous que le fichier contenant la classe RAG s'appelle bien rag.py

app = Flask(__name__)
CORS(app)  # Appliquez CORS à toute l'application Flask
rag = RAG()

@app.route('/')
def home():
    return "Bienvenue sur l'API de Chatbot. Utilisez /ask pour poser des questions."

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()  # Utilise get_json pour les données JSON
    user_message = data.get('messageText', '')
    response = rag.retrieve_response(user_message)
    return jsonify(answer=response)


if __name__ == '__main__':
    app.run(port=5002, debug=True)




