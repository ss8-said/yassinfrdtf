classes = ['specialization_options', 'alumni_success', 'further_learning', 'international_students', 'career_services', 'certification_value', 'program_comparison', 'financial_support', 'application_steps', 'admission_requirements', 'faculty_expertise', 'bot_identity', 'hands_on_projects', 'miscellaneous_questions', 'campus_life', 'error_resolution', 'prerequisite_skills', 'time_commitment', 'tech_requirements', 'greeting', 'core_curriculum', 'ai_definition']
unique_tokens = ['abroad', 'necessitie', 'admission', 'obligatory', 'helper', 'unique', 'first', 'hardware', 'eligibility', 'support', 'advancement', 'hillo', 'educational', 'superior', 'resident', 'commitment', 'differ', 'code', 'hey', 'heyy', 'different', 'good', 'something', 'special', 'recognize', 'research', 'plan', 'pay', 'facility', 'apart', 'credential', 'prep', 'go', 'unrelated', 'skill', 'course', 'knowledge', 'ask', 'week', 'irrelevant', 'need', 'role', 'official', 'unhelpful', 'hi', 'work', 'pick', 'list', 'feature', 'overseas', 'yo', 'world', 'howdy', 'inaccurate', 'daily', 'citizen', 'opportunity', 'set', 'end', 'team', 'tell', 'hello', 'funding', 'admit', 'join', 'new', 'question', 'recognition', 'versus', 'per', 'reduction', 'scholarship', 'kind', 'certify', 'require', 'misunderstood', 'artificial', 'want', 'lecturer', 'learning', 'non', 'job', 'graduation', 'worth', 'statistic', 'sup', 'student', 'mandatory', 'essential', 'alternative', 'must', 'preparation', 'computer', 'matter', 'portfolio', 'achievement', 'native', 'define', 'beginner', 'laptop', 'activity', 'preliminary', 'cost', 'workload', 'would', 'experience', 'point', 'environment', 'pre', 'assistance', 'answer', 'subsequent', 'make', 'address', 'aid', 'option', 'misinterpret', 'present', 'process', 'faculty', 'brief', 'tuition', 'reputation', 'constitute', 'deal', 'payment', 'coursework', 'capability', 'practical', 'search', 'professional', 'comparison', 'curriculum', 'come', 'greet', 'compulsory', 'lifelong', 'placement', 'response', 'principal', 'teach', 'life', 'criterion', 'spec', 'topic', 'education', 'assistant', 'study', 'miss', 'industry', 'change', 'financial', 'chat', 'getting', 'respect', 'enrollment', 'enroll', 'prerequisite', 'identity', 'main', 'fundamental', 'workshop', 'basic', 'foundational', 'ai', 'coding', 'community', 'machine', 'credibility', 'service', 'implementation', 'description', 'helloo', 'friend', 'incorrect', 'allocation', 'learn', 'benefit', 'greeting', 'procedure', 'competitor', 'demand', 'benchmark', 'stand', 'create', 'expectation', 'name', 'direction', 'else', 'success', 'academic', 'competitive', 'specification', 'separate', 'aspect', 'buddy', 'training', 'legitimate', 'deliver', 'another', 'wrong', 'requirement', 'information', 'specialize', 'mark', 'other', 'comparative', 'atmosphere', 'component', 'central', 'expect', 'useful', 'track', 'condition', 'result', 'emphasis', 'destination', 'phase', 'teacher', 'permit', 'next', 'system', 'certification', 'explain', 'ongoing', 'compare', 'edge', 'focus', 'hour', 'timeline', 'hiya', 'completion', 'time', 'relevant', 'term', 'afternoon', 'qualification', 'background', 'know', 'application', 'graduate', 'computing', 'specialty', 'readiness', 'guidance', 'employment', 'follow', 'weekly', 'apply', 'choice', 'investment', 'expertise', 'exist', 'international', 'credible', 'not', 'foreign', 'take', 'human', 'simply', 'switch', 'actual', 'have', 'function', 'culture', 'applicant', 'alumnus', 'concentration', 'purpose', 'technical', 'helpful', 'specialization', 'compute', 'accreditation', 'validation', 'continue', 'building', 'evening', 'explanation', 'morning', 'planning', 'area', 'real', 'path', 'outcome', 'visa', 'value', 'advantage', 'project', 'level', 'vibe', 'factor', 'career', 'available', 'well', 'class', 'choose', 'software', 'instructor', 'hand', 'day', 'get', 'bot', 'campus', 'guide', 'definition', 'technology', 'be', "'s", 'educator', 'subject', 'describe', 'mean', 'market', 'introduce', 'post', 'distinguish', 'preparatory', 'finish', 'advanced', 'program', 'core', 'accredit', 'step', 'professor', 'development', 'prepare', 'prior', 'simple', 'future', 'acceptance', 'intelligence', 'minimum', 'resource', 'inquiry', 'counseling', 'help', 'necessary', 'exactly', 'understanding', 'issue', 'hunting', 'publication', 'degree', 'base', 'entry']
from tensorflow.keras.models import load_model
model = load_model('saidbot.h5')
import numpy as np
import string
import nltk
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import random
import json
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')   
nltk.download('stopwords')
with open("ia_chat_data.json","r",encoding="utf-8") as file :
    chat_data = json.load(file)
word_to_index = {word: i for i, word in enumerate(unique_tokens)}
import numpy as np
import string
from nltk.tokenize import word_tokenize
import random
lemmatizer = WordNetLemmatizer()

def predict_response(sentence):
    # Prétraitement : Tokenisation et lemmatisation
    tokens = word_tokenize(sentence.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in string.punctuation]

    # Transformer la phrase en One-Hot Encoding
    vector = [0] * len(unique_tokens)
    for token in tokens:
        if token in word_to_index:
            vector[word_to_index[token]] = 1  # Activer l'index correspondant

    # Convertir en format compatible avec TensorFlow
    input_data = np.array([vector])

    # Prédiction du modèle
    prediction = model.predict(input_data)
    intent_index = np.argmax(prediction)  # Trouver l'index avec la plus grande probabilité
    predicted_intent = classes[intent_index]  # Récupérer l'intention correspondante
  
    for intent_data in chat_data['intents']:
        if intent_data["tag"] == predicted_intent:
            response = random.choice(intent_data["responses"])  # Choisir une réponse aléatoire
            return response
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI()

templates = Jinja2Templates(directory="public")

class UserInput(BaseModel):
    message: str

# Route pour servir la page HTML
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint d'API
@app.post("/chat")
async def chat(text: UserInput):
    response = predict_response(text.message)
    return {"saidbot": response}