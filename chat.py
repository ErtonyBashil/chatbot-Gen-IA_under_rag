import requests

def chatbot(message):
    url = "http://127.0.0.1:5000/ask"  # URL de l'API Flask
    response = requests.post(url, data={'messageText': message})
    
    if response.status_code == 200:
        return response.json()['answer']
    else:
        return "Erreur lors de la communication avec le serveur."

def main():
    while True:
        user_input = input("Vous: ")
        if user_input.lower() == "exit":
            break  # Sortir de la boucle si l'utilisateur tape "exit"

        response = chatbot(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
