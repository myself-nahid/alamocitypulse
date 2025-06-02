# sentiment_analyzer.py
# Import necessary libraries
import os
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Optional, Literal
from pathlib import Path

# Get the absolute path to the .env file
env_path = Path('.') / '.env'
print(f"Looking for .env file at: {env_path.absolute()}")

# Load environment variables from .env file
load_dotenv(dotenv_path=env_path)

# Set API key directly (for testing)
os.environ["GEMINI_API_KEY"] = "AIzaSyDjvgh9y0adcP8KS9ROor7YEIt2KcaZM8w"  # Replace with your actual API key

# Configure Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
IS_GEMINI_CONFIGURED = False

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        IS_GEMINI_CONFIGURED = True
        # print("Gemini API key configured successfully.")
    except Exception as e: # Broad exception for configuration issues
        print(f"Error configuring Gemini API: {e}")
        print("Please ensure your GEMINI_API_KEY is correct and valid.")
else:
    print("Warning: GEMINI_API_KEY environment variable not set. Sentiment analysis will not function.")

Sentiment = Literal["positive", "negative", "neutral", "error"]

def analyze_news_sentiment(
    title: str,
    description: str,
    image_url: Optional[str] = None,
    link: Optional[str] = None,
    model_name: str = "gemini-1.5-flash"  # Updated to use the recommended model
) -> Sentiment:
    """
    Analyzes the sentiment of a news item using the Google Gemini API.

    Args:
        title (str): The title of the news item.
        description (str): The description or snippet of the news item.
        image_url (Optional[str]): The URL of an associated image (currently not used for sentiment).
        link (Optional[str]): The URL of the news item (currently not used for sentiment).
        model_name (str): The Gemini model to use for the analysis.

    Returns:
        Sentiment: "positive", "negative", "neutral" (if model is unsure or content is neutral),
                   or "error" if an API call fails or an unexpected issue occurs.
    """
    if not IS_GEMINI_CONFIGURED:
        print("Gemini API not configured. Cannot perform sentiment analysis.")
        return "error"

    if not title and not description:
        print("Warning: Title and description are both empty. Returning 'neutral'.")
        return "neutral"

    content_to_analyze = ""
    if description:
        content_to_analyze += f"Description: {description}\n"
    if title:
        content_to_analyze += f"Title: {title}"
    if not description and title:
        content_to_analyze = f"Title: {title}"

    # Combined prompt for Gemini
    full_prompt = (
        "You are a sentiment analysis assistant. "
        "Analyze the sentiment of the provided news item (title and/or description). "
        "Classify it as 'positive', 'negative', or 'neutral'. "
        "Respond with ONLY one of these three words, without any additional explanation or punctuation.\n\n"
        f"News Content:\n{content_to_analyze.strip()}\n\nSentiment:"
    )

    try:
        # List available models for debugging
        # print("Available models:", [model.name for model in genai.list_models()])
        
        generative_model = genai.GenerativeModel(model_name=model_name)
        response = generative_model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=10,
                temperature=0.0
            )
        )

        # Robustly extract text and handle potential blocking
        sentiment_text = ""
        try:
            sentiment_text = response.text.strip().lower()
        except ValueError:
            pass

        if not sentiment_text:
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                print(f"Warning: Content generation blocked by Gemini. Reason: {response.prompt_feedback.block_reason}. Input: '{content_to_analyze[:100]}...'. Defaulting to neutral.")
            else:
                blocked_by_candidate = False
                if response.candidates:
                    for candidate in response.candidates:
                        if candidate.finish_reason.name == "SAFETY":
                            blocked_by_candidate = True
                            print(f"Warning: Content generation blocked by Gemini due to safety settings on candidate. Input: '{content_to_analyze[:100]}...'. Defaulting to neutral.")
                            break
                if not blocked_by_candidate:
                    print(f"Warning: Gemini returned an empty or unexpected response. Input: '{content_to_analyze[:100]}...'. Defaulting to neutral.")
            return "neutral"

        # Validate the response
        if sentiment_text == "positive":
            return "positive"
        elif sentiment_text == "negative":
            return "negative"
        elif sentiment_text == "neutral":
            return "neutral"
        else:
            print(f"Warning: Gemini returned an unexpected sentiment: '{sentiment_text}'. Input: '{content_to_analyze[:100]}...'. Defaulting to neutral.")
            return "neutral"

    except Exception as e:
        print(f"Google Gemini API Error or other exception during sentiment analysis: {e}")
        return "error"

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    if not IS_GEMINI_CONFIGURED:
        print("Gemini API not configured. Skipping examples.")
    else:
        print("\n--- Running Sentiment Analysis Examples with Gemini ---")

        # Example 1: Positive news
        positive_news = {
            "title": "Groundbreaking Discovery Promises Cure for Major Disease",
            "description": "Scientists today announced a revolutionary breakthrough that could lead to a complete cure for a widespread ailment, bringing hope to millions worldwide.",
        }
        sentiment1 = analyze_news_sentiment(**positive_news)
        print(f"News 1 Sentiment: {sentiment1} (Expected: positive)")

        # Example 2: Negative news
        negative_news = {
            "title": "Global Markets Plunge Amidst Economic Uncertainty",
            "description": "Stock markets around the world experienced a sharp decline today as investors reacted to growing fears of an impending recession and geopolitical tensions.",
        }
        sentiment2 = analyze_news_sentiment(**negative_news)
        print(f"News 2 Sentiment: {sentiment2} (Expected: negative)")

        # Example 3: Neutral news / Factual update
        neutral_news = {
            "title": "City Council Announces New Public Transportation Schedule",
            "description": "The city council has released an updated schedule for public bus routes, effective next Monday. Commuters are advised to check the new timings.",
        }
        sentiment3 = analyze_news_sentiment(**neutral_news)
        print(f"News 3 Sentiment: {sentiment3} (Expected: neutral)")

        # Example 4: Only title
        title_only_news = {
            "title": "Company Reports Record Profits This Quarter",
            "description": "",
        }
        sentiment4 = analyze_news_sentiment(**title_only_news)
        print(f"News 4 (Title Only) Sentiment: {sentiment4} (Expected: positive)")

        # Example 5: Ambiguous/Slightly Negative title
        ambiguous_news = {
            "title": "Tech Giant Faces Scrutiny Over Data Privacy Practices",
            "description": "Regulators are launching an investigation into how the company handles user data, raising concerns among privacy advocates.",
        }
        sentiment5 = analyze_news_sentiment(**ambiguous_news)
        print(f"News 5 (Ambiguous/Negative) Sentiment: {sentiment5} (Expected: negative)")

        # Example 6: Potentially positive but could be neutral
        tech_update_news = {
            "title": "New Smartphone Model Unveiled with Enhanced Camera Features",
            "description": "The latest iteration of the popular smartphone boasts an upgraded camera system and faster processing speeds, available for pre-order next week."
        }
        sentiment6 = analyze_news_sentiment(**tech_update_news)
        print(f"News 6 Sentiment: {sentiment6} (Expected: positive or neutral)")

        # Example 7: Empty input
        empty_news = {"title": "", "description": ""}
        sentiment7 = analyze_news_sentiment(**empty_news)
        print(f"News 7 (Empty) Sentiment: {sentiment7} (Expected: neutral)")