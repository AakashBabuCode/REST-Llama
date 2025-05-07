from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from flask import Flask, request, jsonify
import logging
import re
from functools import wraps
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def format_output(text):
    try:
        if not isinstance(text, str):
            logger.warning(f"Non-string input to format_output: {type(text)}")
            return str(text)
        return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    except Exception as e:
        logger.error(f"Error in format_output: {str(e)}")
        return text  # Return original text if formatting fails

def handle_errors(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            logger.error(traceback.format_exc())  # Log full traceback
            return jsonify({
                "error": "Internal server error",
                "details": str(e),
                "endpoint": f.__name__
            }), 500
    return wrapper

def initialize_llama3():
    try:
        logger.info("Initializing chatbot pipeline...")

        create_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are my personal assistant"),
            ("user", "Question: {question}")
        ])

        llama_model = Ollama(
            model="llama3",
            temperature=0.7,
            timeout=30  # 30 seconds timeout
        )
        
        output_parser = StrOutputParser()
        chatbot_pipeline = create_prompt | llama_model | output_parser
        test_response = chatbot_pipeline.invoke({'question': 'Say hello'})
        logger.info(f"Chatbot test response: {test_response}")
        
        return chatbot_pipeline
        
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {str(e)}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Chatbot initialization failed: {str(e)}")

max_retries = 3
retry_delay = 2  
chatbot_pipeline = None

for attempt in range(max_retries):
    try:
        chatbot_pipeline = initialize_llama3()
        break
    except Exception as e:
        if attempt == max_retries - 1:
            logger.critical("Failed to initialize chatbot after multiple attempts")
            raise
        logger.warning(f"Initialization attempt {attempt + 1} failed, retrying...")
        import time
        time.sleep(retry_delay)

@app.route('/api/chat', methods=['POST'])
@handle_errors
def chat():
    if not request.is_json:
        return jsonify({
            "error": "Content-Type must be application/json",
            "status": "error"
        }), 415
    
    data = request.get_json()

    if not data:
        return jsonify({
            "error": "Empty request body",
            "status": "error"
        }), 400
    
    if 'question' not in data:
        return jsonify({
            "error": "Missing 'question' in request body",
            "status": "error"
        }), 400
    
    question = data['question'].strip()
    if not question:
        return jsonify({
            "error": "Question cannot be empty",
            "status": "error"
        }), 400
    
    try:
        logger.info(f"Processing question: {question}")
        response = chatbot_pipeline.invoke({'question': question})
        formatted_response = format_output(response)
        
        return jsonify({
            "response": formatted_response,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Failed to process question",
            "details": str(e),
            "status": "error"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():

    health_status = {
        "service": "chatbot-api",
        "chatbot_initialized": chatbot_pipeline is not None
    }
    
    if chatbot_pipeline is None:
        health_status["status"] = "unhealthy"
        health_status["error"] = "Chatbot pipeline not initialized"
        return jsonify(health_status), 503
    else:
        health_status["status"] = "healthy"
        return jsonify(health_status)

if __name__ == '__main__':
    try:
        logger.info("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}")
        logger.critical(traceback.format_exc())