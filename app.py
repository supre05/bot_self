'''
Main app running backend
'''

# remainig potential optimizations
# cache functions
# backgrnd stt tss

# Import necessary libraries
import os
import csv
import logging
from flask import Flask, render_template, request, jsonify, Response, send_file, send_from_directory
from flask_caching import Cache
from rag_agent import query_agent
from tts_run import get_audio
from stt_tour_workflow import record_audio
from Wake_word import listen_for_wake_word, listen_wake_word_hey_some_vid
from config import exhibit_data_path, output_audio_file, AUDIO_FOLDER

# App Setup
app = Flask(__name__)
app.config['DEBUG'] = os.environ['FLASK_DEBUG']

# Configure logger for debugging
logging.basicConfig(
    level=logging.ERROR,  # Set the minimum severity level to WARNING
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log')  # Write logs to a file named 'app.log'
    ]
)
logger = logging.getLogger(__name__)

# Load exhibit data
with open(exhibit_data_path, "r") as file:
    reader = csv.DictReader(file)
    exhibit_data = list(reader)
    exhibit_dict = {int(exhibit["id"]): exhibit for exhibit in exhibit_data}

# Cache flask for repeated request
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
cache.init_app(app)

# Add browser cache for optimization
@app.after_request
def add_cache_headers(response):
    if request.path.startswith('/static/'):
        response.headers['Cache-Control'] = 'public, max-age=300'
    return response

# Flask routes
@app.route('/')
def home():
    return render_template('home.html', active_page = 'home')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html', active_page = 'chatbot')

@app.route('/about')
def about():
    return render_template('about.html', active_page = 'about')

@app.route('/exhibits')
def exhibits():
    return render_template('exhibits.html', exhibits = exhibit_data, active_page = 'exhibits')

def get_exhibit_by_id(exhibit_id):
    return exhibit_dict.get(exhibit_id + 1)

@app.route('/exhibit/<int:exhibit_id>')
# @cache.cached(timeout=60, query_string=True)
def exhibit_page(exhibit_id):
    exhibit = get_exhibit_by_id(exhibit_id - 1)
    if exhibit:
        return render_template('exhibit.html', exhibit = exhibit, exhibit_id = exhibit_id)
    return "Exhibit not found", 404

@app.route('/tour/<int:current_id>')
# @cache.cached(timeout=60, query_string=True)
def tour(current_id):
    exhibit = get_exhibit_by_id(current_id)
    if exhibit:
        next_id = current_id + 1 if current_id < len(exhibit_data) else None
        prev_id = current_id - 1 if current_id > 1 else None
        return render_template('tour.html', exhibit=exhibit, current_id=current_id, next_id=next_id, prev_id=prev_id)
    return "Exhibit not found", 404


# Audio file handling
if not os.path.exists(AUDIO_FOLDER): os.makedirs(AUDIO_FOLDER)

@app.route('/get', methods=['POST'])
# @cache.cached(timeout=60, query_string=True)  # Cache for 60 seconds
def query(): # Chatbot text interaction
    try:
        user_input = request.form["msg"]
        output = query_agent(user_input) # Chat response
        return jsonify({
            "text": output,
            "audio_url": f"/audio?text={output}"
        })
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify(
            {
            "text": "Sorry, there was an error processing your request.",
            "audio_file": None
        }
        ), 500
    except:
        return None
    
@app.route("/audio")
# @cache.cached(timeout=60, query_string=True)
def audio_file():
    import io
    text = request.args.get("text")

    # Stream audio to the browser
    audio_output = get_audio(text)
    return send_file(audio_output, mimetype="audio/wav", as_attachment=False)
    
@app.route('/stt', methods=['GET'])
def stt(): # Chatbot audio interaction
    transcription = record_audio()
    return jsonify({"text": transcription})

# Wakeword logic 

@app.route('/detect_wake_word', methods=['POST']) 
def detect_wake_word():
    data = request.get_json()
    page = data.get('page') 
    detected_wake_word = listen_for_wake_word()  # Use your actual detection logic

    if detected_wake_word == "yes i do":
        if page == "tour":
            # Logic for "yes i do" on the "tour" page
            #first_question = record_audio()
            return jsonify(action='chatbot')
        elif page == "chatbot":
            wait_for_hey_samvid=listen_wake_word_hey_some_vid()
            if wait_for_hey_samvid == "hey some vid":
                further_question = record_audio()
            return jsonify(action="stt", transcription=further_question)
    
    elif detected_wake_word == "hey some vid":
        question=record_audio()
        return jsonify(action="stt",transcription=question)

    elif detected_wake_word == "no i dont":
        return jsonify(action="next_exhibit", audio="lets_move_on")
    
    else:
        return jsonify(action="ask again", audio="looks_like_no_questions")
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
