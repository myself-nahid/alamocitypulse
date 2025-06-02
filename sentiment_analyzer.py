import os
import openai
from typing import Optional, Literal

# Initialize the OpenAI client
# It's good practice to initialize it once if your application runs continuously,
# but for a standalone function, initializing inside or outside is fine.
# The client will automatically pick up the OPENAI_API_KEY environment variable.
try:
    client = openai.OpenAI()
except openai.OpenAIError as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please ensure your OPENAI_API_KEY environment variable is set correctly.")
    client = None # Or handle as a fatal error depending on your app's needs

Sentiment = Literal["positive", "negative", "neutral", "error"] # Added neutral and error for robustness

def analyze_news_sentiment(
    title: str,
    description: str,
    image_url: Optional[str] = None, # Included as per your spec, though not directly used for text sentiment
    link: Optional[str] = None,       # Included as per your spec, though not directly used for text sentiment
    model: str = "gpt-3.5-turbo" # You can switch to "gpt-4-turbo-preview" or other models
) -> Sentiment:
    """
    Analyzes the sentiment of a news item using the OpenAI API.

    Args:
        title (str): The title of the news item.
        description (str): The description or snippet of the news item.
        image_url (Optional[str]): The URL of an associated image (currently not used for sentiment).
        link (Optional[str]): The URL of the news item (currently not used for sentiment).
        model (str): The OpenAI model to use for the analysis.

    Returns:
        Sentiment: "positive", "negative", "neutral" (if model is unsure or content is neutral),
                   or "error" if an API call fails or an unexpected issue occurs.
    """
    if not client:
        print("OpenAI client not initialized. Cannot perform sentiment analysis.")
        return "error"

    if not title and not description:
        # If both title and description are empty, it's hard to determine sentiment.
        # You might decide to return "neutral" or raise an error.
        print("Warning: Title and description are both empty. Returning 'neutral'.")
        return "neutral"

    # Combine title and description for a more comprehensive analysis.
    # Prioritize description if available, as it usually contains more context.
    content_to_analyze = ""
    if description:
        content_to_analyze += f"Description: {description}\n"
    if title: # Add title, even if description is present, for full context.
        content_to_analyze += f"Title: {title}"

    # If only title is present
    if not description and title:
        content_to_analyze = f"Title: {title}"
    
    # System prompt to guide the AI
    system_prompt = (
        "You are a sentiment analysis assistant. "
        "Analyze the sentiment of the provided news item (title and/or description). "
        "Classify it as 'positive', 'negative', or 'neutral'. "
        "Respond with ONLY one of these three words, without any additional explanation or punctuation."
    )

    # User prompt with the content
    user_prompt = f"News Content:\n{content_to_analyze.strip()}\n\nSentiment:"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,  # Low temperature for classification tasks
            max_tokens=10,    # "positive", "negative", "neutral" are short
            n=1,
            stop=None
        )

        sentiment_text = response.choices[0].message.content.strip().lower()

        # Validate the response
        if sentiment_text == "positive":
            return "positive"
        elif sentiment_text == "negative":
            return "negative"
        elif sentiment_text == "neutral":
            return "neutral"
        else:
            print(f"Warning: OpenAI returned an unexpected sentiment: '{sentiment_text}'. Input: '{content_to_analyze[:100]}...'. Defaulting to neutral.")
            # You might want to log this for review
            return "neutral" # Or "error" if you want to be stricter

    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
        return "error"
    except Exception as e:
        print(f"An unexpected error occurred during sentiment analysis: {e}")
        return "error"

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set. Please set it to run examples.")
    else:
        print("OpenAI client initialized successfully for testing.")
        
        # Example 1: Positive news
        positive_news = {
            "title": "Groundbreaking Discovery Promises Cure for Major Disease",
            "description": "Scientists today announced a revolutionary breakthrough that could lead to a complete cure for a widespread ailment, bringing hope to millions worldwide.",
            "image_url": "http://example.com/image_positive.jpg",
            "link": "http://example.com/news_positive"
        }
        sentiment1 = analyze_news_sentiment(**positive_news)
        print(f"News 1 Sentiment: {sentiment1} (Expected: positive)")

        # Example 2: Negative news
        negative_news = {
            "title": "Global Markets Plunge Amidst Economic Uncertainty",
            "description": "Stock markets around the world experienced a sharp decline today as investors reacted to growing fears of an impending recession and geopolitical tensions.",
            "image_url": "http://example.com/image_negative.jpg",
            "link": "http://example.com/news_negative"
        }
        sentiment2 = analyze_news_sentiment(**negative_news)
        print(f"News 2 Sentiment: {sentiment2} (Expected: negative)")

        # Example 3: Neutral news / Factual update
        neutral_news = {
            "title": "City Council Announces New Public Transportation Schedule",
            "description": "The city council has released an updated schedule for public bus routes, effective next Monday. Commuters are advised to check the new timings.",
            "image_url": "http://example.com/image_neutral.jpg",
            "link": "http://example.com/news_neutral"
        }
        sentiment3 = analyze_news_sentiment(**neutral_news)
        print(f"News 3 Sentiment: {sentiment3} (Expected: neutral or positive/negative depending on interpretation)")
        
        # Example 4: Only title
        title_only_news = {
            "title": "Company Reports Record Profits This Quarter",
            "description": "", # Empty description
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