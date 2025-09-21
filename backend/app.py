from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import requests
from bs4 import BeautifulSoup
import logging
import traceback

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global detector variable
detector = None

def init_model():
    """Initialize the ML model with fallbacks"""
    global detector
    
    try:
        from transformers import pipeline
        
        # Try fake news model first
        logger.info("Attempting to load fake news detection model...")
        detector = pipeline(
            "text-classification",
            model="hamzab/roberta-fake-news-classification",
            top_k=None  # Updated parameter instead of return_all_scores
        )
        logger.info("✅ Successfully loaded RoBERTa fake news model")
        return "fake_news"
        
    except Exception as e:
        logger.warning(f"⚠️  Fake news model failed: {e}")
        
        try:
            # Fallback to sentiment model
            logger.info("Falling back to sentiment analysis model...")
            detector = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                top_k=None
            )
            logger.info("✅ Successfully loaded sentiment model as fallback")
            return "sentiment"
            
        except Exception as e:
            logger.warning(f"⚠️  Sentiment model failed: {e}")
            
            try:
                # Final fallback to basic model
                logger.info("Using basic distilbert model...")
                detector = pipeline(
                    "text-classification",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    top_k=None
                )
                logger.info("✅ Successfully loaded basic model")
                return "basic"
                
            except Exception as e:
                logger.error(f"❌ All models failed: {e}")
                return None

def init_db():
    """Initialize database"""
    try:
        conn = sqlite3.connect('misinformation.db')
        conn.execute('''
          CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_type TEXT, 
            input_text TEXT,
            raw_label TEXT,
            interpreted_label TEXT,
            confidence_score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
          ''')
        conn.commit()
        conn.close()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")

def extract_text_from_url(url):
    """Extract text from URL with basic error handling"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        logger.info(f"Fetching URL: {url}")
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'aside', 'header']):
            element.decompose()
        
        # Try to find main content
        content = ""
        for selector in ['article', '[role="article"]', '.content', '.post', 'main', 'body']:
            elements = soup.select(selector)
            if elements:
                content = elements[0].get_text(separator=' ', strip=True)
                if len(content) > 100:  # Only use if substantial content
                    break
        
        if not content:
            content = soup.get_text(separator=' ', strip=True)
        
        # Clean and limit content
        content = ' '.join(content.split())
        return content[:1500] if len(content) > 1500 else content
        
    except Exception as e:
        logger.error(f"URL extraction failed: {e}")
        raise Exception(f"Could not fetch content from URL: {str(e)}")

def analyze_content_smart(content, model_type):
    """Smart content analysis with pre-filtering"""
    
    # Pre-filter obvious factual statements
    factual_indicators = [
        'is a', 'is an', 'are a', 'are an', 'is the', 'are the',
        'definition', 'means', 'refers to', 'known as'
    ]
    
    simple_facts = [
        'fruit', 'vegetable', 'animal', 'plant', 'color', 'number',
        'country', 'city', 'mountain', 'river', 'ocean'
    ]
    
    content_lower = content.lower()
    
    # Check if it's a simple factual statement
    is_simple_fact = (
        len(content.split()) < 20 and  # Short statement
        any(indicator in content_lower for indicator in factual_indicators) and
        any(fact in content_lower for fact in simple_facts)
    )
    
    if is_simple_fact:
        return {
            'label': 'FACTUAL',
            'score': 0.95,
            'note': 'Simple factual statement detected'
        }
    
    # Check content length and complexity
    if len(content.split()) < 10:
        return {
            'label': 'TOO_SHORT',
            'score': 0.5,
            'note': 'Content too short for reliable analysis'
        }
    
    # Run ML model for substantial content
    try:
        logger.info(f"Running ML model on content: {content[:100]}...")
        analysis = detector(content)
        logger.info(f"Raw ML output: {analysis}")
        
        # Handle different output formats
        if isinstance(analysis, list) and len(analysis) > 0:
            if isinstance(analysis[0], list):
                # Format: [[{'label': 'X', 'score': Y}, ...]]
                best = max(analysis[0], key=lambda x: x['score'])
            else:
                # Format: [{'label': 'X', 'score': Y}, ...]
                best = max(analysis, key=lambda x: x['score'])
        else:
            # Unexpected format
            logger.warning(f"Unexpected analysis format: {type(analysis)}")
            best = {'label': 'Unknown', 'score': 0.5}
        
        logger.info(f"Best prediction: {best}")
        
        return {
            'label': best['label'],
            'score': best['score'],
            'note': 'ML model analysis'
        }
        
    except Exception as e:
        logger.error(f"ML analysis failed: {e}")
        logger.error(f"ML analysis traceback: {traceback.format_exc()}")
        return {
            'label': 'ERROR',
            'score': 0.5,
            'note': f'Analysis failed: {str(e)}'
        }

def interpret_result(label, score, model_type="fake_news", note=""):
    """Enhanced result interpretation with context"""
    
    try:
        # Handle special cases first
        if label == 'FACTUAL':
            return "Factual Statement", "✅"
        elif label == 'TOO_SHORT':
            return "Content Too Brief to Analyze", "⚠️"
        elif label == 'ERROR':
            return "Analysis Error", "❌"
        
        # Handle model-specific results
        if model_type == "fake_news":
            if label.upper() == "REAL" or label.upper() == "TRUE":
                if score > 0.9:
                    return "Highly Reliable Content", "✅"
                elif score > 0.7:
                    return "Likely Reliable Content", "✅"
                elif score > 0.6:
                    return "Probably Reliable Content", "✅"
                else:
                    return "Possibly Reliable Content", "⚠️"
            elif label.upper() == "FAKE" or label.upper() == "FALSE":
                # Be more conservative with fake classifications for simple content
                if score > 0.95:
                    return "Potentially Misleading Content", "⚠️"
                elif score > 0.8:
                    return "Questionable Content", "⚠️"  
                else:
                    return "Requires Further Verification", "⚠️"
        
        elif model_type == "sentiment":
            if label.upper() == "POSITIVE":
                return "Positive Sentiment Content", "✅"
            elif label.upper() == "NEGATIVE": 
                return "Critical/Negative Content", "⚠️"
            else:
                return "Neutral Sentiment Content", "⚠️"
        
        # Default fallback for any unhandled cases
        return f"Classification: {label}", "⚠️"
        
    except Exception as e:
        logger.error(f"Result interpretation error: {e}")
        return f"Classification: {label}", "⚠️"

# Initialize everything
logger.info("🚀 Starting TruthGuard backend...")
model_type = init_model()
init_db()

if detector is None:
    logger.error("❌ No model could be loaded! App will not work properly.")
else:
    logger.info(f"✅ TruthGuard backend ready with {model_type} model!")

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        logger.info("📝 New analysis request received")
        
        # Check if model is loaded
        if detector is None:
            return jsonify(error="ML model not available. Please check server logs."), 500
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify(error="No JSON data provided"), 400
        
        text = data.get('text', '').strip()
        url = data.get('url', '').strip()
        
        logger.info(f"Request data - Text length: {len(text)}, URL: {bool(url)}")
        
        if not text and not url:
            return jsonify(error="Please provide text or URL to analyze"), 400
        
        # Extract content
        if url:
            try:
                content = extract_text_from_url(url)
                input_type = "url"
                input_text = url
                logger.info(f"✅ URL content extracted: {len(content)} characters")
            except Exception as e:
                logger.error(f"URL processing failed: {e}")
                return jsonify(error=str(e)), 400
        else:
            content = text[:1500]  # Limit text length
            input_type = "text"
            input_text = text
            logger.info(f"✅ Text input received: {len(content)} characters")
        
        if not content.strip():
            return jsonify(error="No meaningful content found to analyze"), 400
        
        # Run smart analysis
        logger.info("🤖 Running smart content analysis...")
        analysis_result = analyze_content_smart(content, model_type)
        
        raw_label = analysis_result['label']
        confidence = float(analysis_result['score'])
        analysis_note = analysis_result.get('note', '')
        
        logger.info(f"Analysis result: {raw_label} ({confidence:.3f}) - {analysis_note}")
        
        # Interpret results with context
        interpreted_label, icon = interpret_result(raw_label, confidence, model_type, analysis_note)
        
        # Store in database (with error handling)
        try:
            conn = sqlite3.connect('misinformation.db')
            conn.execute(
                """INSERT INTO analyses 
                   (input_type, input_text, raw_label, interpreted_label, confidence_score) 
                   VALUES (?, ?, ?, ?, ?)""",
                (input_type, input_text[:500], raw_label, interpreted_label, confidence)
            )
            conn.commit()
            conn.close()
            logger.info("✅ Result stored in database")
        except Exception as e:
            logger.warning(f"⚠️  Database storage failed: {e}")
            # Continue anyway - don't fail the request
        
        # Prepare response with additional context
        response_data = {
            'label': raw_label,
            'interpreted_label': interpreted_label,
            'score': round(confidence, 3),
            'confidence_percentage': round(confidence * 100, 1),
            'icon': icon,
            'content_preview': content[:200] + "..." if len(content) > 200 else content,
            'model_type': model_type,
            'analysis_note': analysis_note,
            'content_length': len(content),
            'word_count': len(content.split())
        }
        
        logger.info(f"✅ Analysis complete: {interpreted_label}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return jsonify(error=f"Analysis failed: {str(e)}"), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector is not None,
        'model_type': model_type if detector else None
    })

@app.route('/', methods=['GET'])
def home():
    """Simple home endpoint"""
    return jsonify({'message': 'TruthGuard API is running!', 'status': 'ok'})

if __name__ == '__main__':
    print("🚀 Starting TruthGuard Flask server...")
    print("🌐 Access at: http://localhost:5000")
    print("🔍 Health check: http://localhost:5000/api/health")
    app.run(host='0.0.0.0', port=5000, debug=True)