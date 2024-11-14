'''
Main app running backend
'''

# Import necessary libraries
import os
import csv
import logging
from flask import Flask, render_template, request, jsonify
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
    level=logging.WARNING,  # Set the minimum severity level to WARNING
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Print logs to console
        logging.FileHandler('app.log')  # Write logs to a file named 'app.log'
    ]
)
logger = logging.getLogger(__name__)

# Load exhibit data
with open(exhibit_data_path, "r") as file:
    reader = csv.DictReader(file)
    exhibit_data = list(reader)

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
    return exhibit_data[exhibit_id - 1]

@app.route('/exhibit/<int:exhibit_id>')
def exhibit_page(exhibit_id):
    exhibit = get_exhibit_by_id(exhibit_id)
    if exhibit:
        return render_template('exhibit.html', exhibit = exhibit, exhibit_id = exhibit_id)
    return "Exhibit not found", 404

@app.route('/tour/<int:current_id>')
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
def query(): # Chatbot text interaction
    try:
        user_input = request.form["msg"]
        output = query_agent(user_input) # Chat response
        output_audio = get_audio(output) # TTS

        # On dev:
        # Use output_audio vs file. fix bugs
        
        # Return both text and audio file path
        return jsonify({
            "text": output,
            "audio_file": output_audio_file  # Return relative path #output_audio_file
        })
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify(
            {
            "text": "Sorry, there was an error processing your request.",
            "audio_path": None
        }
        ), 500


@app.route('/stt', methods=['GET'])
def stt(): # Chatbot audio interaction
    transcription = record_audio()
    return jsonify({"text": transcription})

# Wakeword logic (pending)

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
    

    # if detected_wake_word == "hey some vid":
    #     stt_text = record_audio()
    #     return jsonify(action="stt", transcription=stt_text)
    
    # elif detected_wake_word == "yes i do":
    #     return jsonify(action="chatbot")
    elif detected_wake_word == "hey some vid":
        question=record_audio()
        return jsonify(action="stt",transcription=question)

    elif detected_wake_word == "no i dont":
#         #main("Let's move on.")
        return jsonify(action="next_exhibit", audio="lets_move_on")
    
    else:
# #         #main("Shall we move on?") 
# #         #time.sleep(3)  # wait for 3 seconds
        return jsonify(action="ask again", audio="looks_like_no_questions")
    



if __name__ == '__main__':
    app.run(debug = app.config['DEBUG'])