import logging
logging.basicConfig(
    filename='review_analyzer.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'  # -> only write important stuff (info and more)
)
from typing import Dict, List
import json
class APIClient:  # -> this class talks to hugging face to check if reviws are happy or sad
    """Handles API requests to Hugging Face for sentiment analysis."""

    def __init__(self, config_path: str = 'config.json'):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.api_key = config.get('huggingface_api_key')
                if not self.api_key:
                    raise ValueError("API key not found in config.json")
            self.api_url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
            self.headers = {"Authorization": f"Bearer {self.api_key}"}
            logging.info("APIClient initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize APIClient: {e}")  # -> write the mistake in diary
            raise

    import time
    import pandas as pd
    def send_request(self, review_text: str) -> Dict:  # -> this sends a review to hugging face and gets answer
        """Sends a request to Hugging Face API and returns sentiment."""

        cache = self._load_cache()  # -> check if we alredy have the answer saved
        cache_key = review_text.lower().strip()
        if cache_key in cache:
            logging.info("Cache hit for review")  # -> write in diary we found it in cache
            return cache[cache_key]

        try:  # -> try to send to hugging face
            payload = {"inputs": review_text}
            response = requests.post(self.api_url, headers=self.headers,
                                     json=payload)  # -> send the reviev to hugging face
            response.raise_for_status()  # -> check if hugging face sent back a good answer
            result = response.json()[0]  # -> get the answer from hugging face
            sentiment = 'positive' if result[0][
                                          'label'] == 'POSITIVE' else 'negative'  # -> decide if review is happy or sad
            score = result[0]['score']  # -> get the score (how sure hugging face is)
            # Simplified key points extraction (Hugging Face doesn't provide this, so we mock it)
            key_points = [f"Sentiment score: {score:.2f}"]  # -> make a fake point with the score
            output = {"sentiment": sentiment, "key_points": key_points}

            # Save to cache
            cache[cache_key] = output
            self._save_cache(cache)
            logging.info("API request successful")  # -> write in diary that we got answer
            return output
        except Exception as e:
            logging.error(f"API request failed: {e}")  # -> write the mistake in diary if something is wrong
            if "429" in str(e):  # -> if hugging face says we sent too many (rate limit)
                time.sleep(5)  # -> wait 5 seconds
                return self.send_request(review_text)  # -> try again
            raise  # -> stop if other error happens

    def _load_cache(self) -> Dict:  # -> this loads the saved answers from a file
        """Loads cached API responses from file."""
        try:
            if os.path.exists('cache.json'):
                with open('cache.json', 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logging.error(f"Failed to load cache: {e}")
            return {}

    from jinja2 import Environment, FileSystemLoader
    def _save_cache(self, cache: Dict) -> None:  # -> this saves the answers to a file
        """Saves cache to file."""
        try:  # -> try to save
            with open('cache.json', 'w') as f:
                json.dump(cache, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save cache: {e}")  # -> write mistake in diary if something gets failed
import os
import requests
class DataLoader:
    """Loads and preprocesses customer review data."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None  # -> make a place to save reveiws later
        logging.info(f"DataLoader initialized with file: {file_path}")

    def load_data(self) -> pd.DataFrame:
        """Loads reviews from a CSV file."""
        try:
            self.data = pd.read_csv(self.file_path)
            self.data.dropna(subset=['review'], inplace=True)  # -> remove any empty reviews
            logging.info(f"Loaded {len(self.data)} reviews")  # -> write in diary how many reviews we got
            return self.data
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise


class ReviewAnalyzer:  # -> this class checks if reviews are happy or sad
    """Analyzes customer reviews using Hugging Face API."""

    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        logging.info("ReviewAnalyzer initialized")

    def analyze_review(self, review_text: str) -> Dict:
        """Analyzes a single review."""
        try:
            result = self.api_client.send_request(review_text)  # -> send review to hugging face
            if result['key_points'][0].startswith("Sentiment score: 0.5") or \
                    float(result['key_points'][0].split(": ")[1]) < 0.6:  # -> if score is low, say its neutral
                result['sentiment'] = 'neutral'
            logging.info(f"Analyzed review: {review_text[:50]}...")  # -> write in diary we checked it
            return result
        except Exception as e:
            logging.error(f"Failed to analyze review: {e}")
            return {"sentiment": "error", "key_points": [str(e)]}


class ReportGenerator:
    """Generates an HTML report with reviews."""

    def __init__(self, output_path: str = 'report.html'):
        self.output_path = output_path
        self.env = Environment(loader=FileSystemLoader('templates'))  # -> set up to use html template
        logging.info(f"ReportGenerator initialized with output: {output_path}")

    def generate_report(self, analysis_results: List[Dict], reviews: pd.DataFrame) -> None:
        """Generates an HTML report."""
        try:
            report_data = []
            for idx, result in enumerate(analysis_results):  # -> go through each reveiw result
                report_data.append({
                    'review_id': reviews.iloc[idx]['id'],
                    'review_text': reviews.iloc[idx]['review'],
                    'sentiment': result.get('sentiment', 'unknown'),
                    'key_points': '; '.join(result.get('key_points', []))
                })

            # Calculate summary
            report_df = pd.DataFrame(report_data)
            sentiment_counts = report_df['sentiment'].value_counts(
                normalize=True) * 100
            summary = {
                'total_reviews': len(report_df),  # -> total revievs
                'positive_percentage': round(sentiment_counts.get('positive', 0), 2),
                'negative_percentage': round(sentiment_counts.get('negative', 0), 2),
                'neutral_percentage': round(sentiment_counts.get('neutral', 0), 2)
            }
            logging.info(f"Report summary: {summary}")

            # Render HTML report
            template = self.env.get_template('report.html')
            with open(self.output_path, 'w') as f:
                f.write(template.render(results=report_data, summary=summary))
            logging.info(f"Report saved to {self.output_path}")
        except Exception as e:
            logging.error(f"Failed to generate report: {e}")
            raise


from flask import Flask, request, render_template
app = Flask(__name__)  # -> start the web app with flask


@app.route('/', methods=['GET', 'POST'])  # -> makes the web page
def index():
    if request.method == 'POST':
        try:
            # Handle file upload
            file = request.files['file']
            if not file or not file.filename.endswith('.csv'):
                return "Please upload a valid CSV file.", 400

            file_path = 'uploaded_reviews.csv'
            file.save(file_path)

            # Initialize components
            api_client = APIClient()
            data_loader = DataLoader(file_path)
            analyzer = ReviewAnalyzer(api_client)
            report_generator = ReportGenerator()

            # Process reviews
            reviews = data_loader.load_data()
            analysis_results = [analyzer.analyze_review(review) for review in reviews['review']]
            report_generator.generate_report(analysis_results, reviews)

            return app.send_static_file('report.html')
        except Exception as e:
            logging.error(f"Web app error: {e}")
            return f"An error occurred: {e}", 500


if __name__ == '__main__':
    app.run(debug=True)