import configparser
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Charger les variables d'environnement
load_dotenv()

class RAG:
    def __init__(self):
        # Chargement de la configuration
        self.CONFIG_FILE = "RAG.ini"
        self.conf = configparser.ConfigParser()
        self.conf.read(self.CONFIG_FILE)

        # Modèle et configuration
        self.embeddings_model = self.conf['RAG']['embedding_model']
        self.llm_model_path = self.conf['RAG']['llm_model_path']
        self.persist_directory = self.conf['RAG']['persist_directory']
        self.cache_folder = self.conf['RAG']['cache_folder']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Utilisation de : {self.device}")

        # Initialisation des embeddings et du modèle
        print("Chargement des embeddings...")
        self.embedder = HuggingFaceEmbeddings(
            model_name=self.embeddings_model,
            cache_folder=self.cache_folder,
            model_kwargs={'device': self.device}
        )
        print("Embeddings chargés.")

        print("Chargement du modèle de langage...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.llm_model_path, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32)
        self.model.to(self.device)
        print("Modèle de langage chargé.")

        # Initialisation de Chroma pour les recherches
        self.vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedder)

    def generate_response(self, prompt_text, max_new_tokens=200):
        # Préparation du texte d'entrée pour le modèle
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        
        # Génération de réponse avec longueur stricte
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        
        # Décodage de la réponse générée
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Élimination des lignes redondantes
        unique_lines = []
        for line in response_text.splitlines():
            if line not in unique_lines:
                unique_lines.append(line)
        
        return "\n".join(unique_lines)

    def similarity_search(self, query, k=5, threshold=0.3):
        # Recherche de documents similaires et vérification de la similarité
        results = self.vectordb.similarity_search_with_score(query, k=k)
        filtered_results = [doc for doc, score in results if score >= threshold]
        return filtered_results

    def retrieve_response(self, query):
        # Template de prompt pour précision
        template = """
        Utilisez uniquement les informations du contexte fourni pour répondre de façon concise à la question suivante. 
        Si la réponse n'est pas dans le contexte, répondez simplement : "Désolé, je n'ai pas trouvé de réponse spécifique à votre question."

        Contexte : {context}
        ---
        Question : {question}
        Réponse :
        """

        # Recherche du contexte pertinent
        print(f"Question posée : {query}")
        matched_docs = self.similarity_search(query, k=5)
        
        # Vérification si des documents similaires ont été trouvés
        if not matched_docs:
            return "Désolé, je n'ai pas trouvé de réponse spécifique à votre question."

        context = "\n".join([doc.page_content for doc in matched_docs])

        # Création du prompt
        prompt = template.format(context=context, question=query)
        print("Génération de la réponse...")

        # Génération de la réponse
        response = self.generate_response(prompt, max_new_tokens=200)
        
        # Nettoyage de la réponse
        response = response.split("Réponse :", 1)[-1].strip()
        
        return response
