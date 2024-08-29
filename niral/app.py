from flask import Flask, request, render_template, redirect, url_for
import os
import spacy
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from collections import Counter, defaultdict

app = Flask(__name__)

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Define stop words for filtering
nltk_stop_words = set(stopwords.words('english'))
spacy_stop_words = nlp.Defaults.stop_words

# Path to downloaded files
directory_path = './text_files/'

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def detect_fuel_type(text):
    lower_text = text.lower()
    fuel_type = 'null'
    mentions_petrol = 'petrol' in lower_text
    mentions_diesel = 'diesel' in lower_text

    if mentions_petrol and mentions_diesel:
        fuel_type = 'petrol/diesel'
    elif mentions_petrol:
        fuel_type = 'petrol'
    elif mentions_diesel:
        fuel_type = 'diesel'

    return fuel_type

def detect_transmission_intent(text):
    lower_text = text.lower()
    transmission_type = 'null'
    mentions_manual = 'manual' in lower_text
    mentions_automatic = 'automatic' in lower_text

    if mentions_manual and mentions_automatic:
        transmission_type = 'both'
    elif mentions_manual:
        transmission_type = 'manual'
    elif mentions_automatic:
        transmission_type = 'automatic'

    return transmission_type

def detect_preferred_color(text):
    lower_text = text.lower()
    color_counter = Counter()

    common_colors = [
        'red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'pink',
        'purple', 'brown', 'grey', 'cyan', 'magenta', 'violet', 'indigo', 'silver','gray'
    ]

    for color in common_colors:
        if color in lower_text:
            color_counter[color] += lower_text.count(color)

    if color_counter:
        most_common_color, _ = color_counter.most_common(1)[0]
        return most_common_color
    return 'none'

def detect_car_type(text):
    car_types = [
        'Sedan', 'Hatchback', 'SUV', 'Crossover', 'Coupe', 'Convertible',
        'Wagon', 'Pickup Truck', 'Minivan', 'Sports Car', 'Luxury Car',
        'Hybrid Car', 'Electric Car', 'Plug-in Hybrid Electric Vehicle', 'Off-Road Vehicle'
    ]

    lower_text = text.lower()
    detected_types = [car_type for car_type in car_types if car_type.lower() in lower_text]
    return detected_types if detected_types else ['none']

def extract_years(text, doc):
    years = set()
    for ent in doc.ents:
        if ent.label_ == 'DATE' and len(ent.text) == 4 and ent.text.isdigit():
            years.add(ent.text)
    return list(years) if years else ['none']

def extract_distance_quantities(text, doc):
    distances = set()
    for ent in doc.ents:
        if ent.label_ == 'QUANTITY':
            text = ent.text.lower()
            if any(unit in text for unit in ['km', 'kilometers', 'miles', 'meters']):
                distances.add(ent.text)
    return list(distances) if distances else ['none']

def detect_feature_with_sentiment(text, feature_keyword):
    lower_text = text.lower()
    blob = TextBlob(lower_text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0 and feature_keyword.lower() in lower_text:
        return True
    return False

def process_text_if_positive_sentiment(text, filename):
    blob = TextBlob(text.lower())
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        doc = nlp(text)

        fuel_type = detect_fuel_type(text)
        transmission_type = detect_transmission_intent(text)
        preferred_color = detect_preferred_color(text)
        car_types_detected = detect_car_type(text)
        models_found = [entity.text for entity in doc.ents if entity.label_ == 'PRODUCT']
        years_found = extract_years(text, doc)
        distances_found = extract_distance_quantities(text, doc)

        free_rc_transfer = detect_feature_with_sentiment(text, 'Register')
        money_back_guarantee = detect_feature_with_sentiment(text, 'Guarantee')
        free_rsa = detect_feature_with_sentiment(text, 'Insurance')
        return_policy = detect_feature_with_sentiment(text, 'Return')

        return {
            "filename": filename,
            "fuel_type": fuel_type,
            "transmission_type": transmission_type,
            "preferred_color": preferred_color,
            "car_types": car_types_detected,
            "models": models_found if models_found else "None",
            "years": years_found,
            "distances": distances_found,
            "free_rc_transfer": free_rc_transfer,
            "money_back_guarantee": money_back_guarantee,
            "free_rsa": free_rsa,
            "return_policy": return_policy
        }
    else:
        return None

def detect_negative_sentences(text):
    sentences = sent_tokenize(text)
    negative_sentences = []

    for sentence in sentences:
        sentiment_score = sia.polarity_scores(sentence)
        if sentiment_score['compound'] < -0.1:
            negative_sentences.append(sentence)

    return negative_sentences

def summarize_feedback(feedback):
    categories = {
        "Refurbishment Quality": ["refurbishment", "quality", "repair", "renovation", "defects", "flaws", "workmanship", "materials", "paint", "job", "standards"],
        "Car Issues": ["engine", "transmission", "brakes", "battery", "suspension", "performance", "electronic", "fuel system", "alignment", "power steering", "exhaust", "vibration", "infotainment", "wipers"],
        "Price Issues": ["price", "cost", "expensive", "value", "fees", "overpriced", "bill", "repair costs", "pricing", "charges"],
        "Customer Experience Issues": ["salesperson", "customer service", "support", "staff", "service center", "communication", "wait time", "appointment", "process", "experience", "staff attitude", "facilities", "follow-up", "instructions", "delays"]
    }

    summarized_feedback = defaultdict(list)

    for feedback_item in feedback:
        feedback_item_lower = feedback_item.lower()
        categorized = False
        for category, keywords in categories.items():
            if any(keyword in feedback_item_lower for keyword in keywords):
                summarized_feedback[category].append(feedback_item)
                categorized = True
        if not categorized:
            summarized_feedback["Uncategorized"].append(feedback_item)

    summary = ""
    for category, items in summarized_feedback.items():
        if items:
            summary += f"- {category}:\n"
            for item in items:
                summary += f"  - {item}\n"

    return summary

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload_files', methods=['POST'])
def upload_files():
    if 'file' not in request.files:
        return redirect(url_for('upload'))

    files = request.files.getlist('file')
    results = []

    for file in files:
        if file.filename == '':
            continue

        if file and file.filename.endswith('.txt'):
            filepath = os.path.join(directory_path, file.filename)
            file.save(filepath)

            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            # Process the text
            positive_results = process_text_if_positive_sentiment(text, file.filename)
            negative_sentences = detect_negative_sentences(text)
            feedback_summary = summarize_feedback(negative_sentences) if negative_sentences else "No negative feedback detected."

            results.append({
                "filename": file.filename,
                "positive_results": positive_results,
                "feedback_summary": feedback_summary
            })

    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
