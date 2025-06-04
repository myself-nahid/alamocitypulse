import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY   # use your own API key
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()

client = OpenAI() # Will now use OPENAI_API_KEY from .env file

# Configure basic logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Function for OpenAI API Call ---
def get_openai_sentiment(text_content: str, model: str = "gpt-4o-mini", temperature: float = 0.1) -> str:
    """
    Calls the OpenAI API to get sentiment for the given text.

    Args:
        text_content (str): The text to analyze.
        model (str): The OpenAI model to use.
        temperature (float): The sampling temperature for the model.

    Returns:
        str: "positive", "negative", or "neutral" (as a fallback if parsing fails).
    """
    if not text_content or text_content.isspace():
        logging.warning("Received empty text_content for sentiment analysis.")
        return "neutral" # Cannot determine sentiment for empty text

    system_prompt = (
        "You are a sentiment analysis expert. Your task is to classify the sentiment "
        "of the provided news article content as either 'positive' or 'negative'. "
        "Consider the overall emotional tone and impact of the news. "
        "Respond with ONLY the word 'positive' or 'negative'."
    )

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_content}
            ],
            temperature=temperature,
            max_tokens=10  # "positive" or "negative" is short
        )
        sentiment = completion.choices[0].message.content.strip().lower()

        if sentiment in ["positive", "negative"]:
            return sentiment
        else:
            # This case handles if the model returns something unexpected
            logging.warning(f"OpenAI returned an unexpected sentiment: '{sentiment}'. Input: '{text_content[:100]}...'")
            
            if "positive" in sentiment: return "positive" # Basic guard
            if "negative" in sentiment: return "negative" # Basic guard
            return "neutral" # Fallback if not strictly "positive" or "negative"

    except Exception as e:
        logging.error(f"Error calling OpenAI API for sentiment: {e}")
        return "error" # Indicates an API call or processing error

# --- Main Sentiment Analysis Function ---
def analyze_news_item_sentiment(news_data: dict) -> str:
    """
    Analyzes the sentiment of a news item based on its fields.

    Args:
        news_data (dict): A dictionary containing news item fields:
            'Category' (str, optional)
            'URL' (str, optional but good for logging)
            'Headline' (str)
            'Image' (str, optional)
            'Description' (str, optional)
            'Published Datetime' (str, optional)
            'Published Text' (str, optional) - This field from your example seems to be
                                            just a formatted datetime, not the article body.
                                            If it *were* the full article text, it would be primary.

    Returns:
        str: The predicted sentiment ("positive", "negative", "neutral" or "error").
    """
    if not isinstance(news_data, dict):
        logging.error("Invalid input: news_data must be a dictionary.")
        return "error"

    headline = news_data.get("Headline", "")
    description = news_data.get("Description", "")
    
    text_for_analysis = []
    if headline:
        text_for_analysis.append(f"Headline: {headline}")
    if description:
        # Limit description length if it's extremely long to manage token usage
        # For GPT-3.5-turbo, context window is ~4k tokens, GPT-4 ~8k or ~32k.
        # A typical word is ~1.3 tokens.
        max_desc_chars = 3000 # Approx 750 words / 1000 tokens, adjustable
        truncated_description = description[:max_desc_chars]
        if len(description) > max_desc_chars:
            truncated_description += "..."
        text_for_analysis.append(f"Description: {truncated_description}")

    if not text_for_analysis:
        logging.warning(f"No text content (headline/description) found for sentiment analysis. URL: {news_data.get('URL', 'N/A')}")
        return "neutral" # Or "error" depending on how you want to handle this

    combined_text = "\n\n".join(text_for_analysis)

    logging.info(f"Analyzing sentiment for URL: {news_data.get('URL', 'N/A')}")
    sentiment = get_openai_sentiment(combined_text)
    logging.info(f"Predicted sentiment: {sentiment} for URL: {news_data.get('URL', 'N/A')}")

    return sentiment

# --- Example Usage with your provided data ---
if __name__ == "__main__":
    # Ensure your OPENAI_API_KEY is set in your environment
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it before running the script, e.g.:")
        print("export OPENAI_API_KEY='your_key_here'")
        exit()

    sample_news_items = [
        {
            "Category": "News",
            "URL": "https://www.ksat.com/sports/2013/08/27/download-the-big-game-coverage-app-from-ksat/",
            "Headline": "Download KSAT's Big Game Coverage App to get real-time scores, highlights, schedules and eSports",
            "Image": "https://res.cloudinary.com/...",
            "Description": "Kolten Parker, Manager of Content and Coverage Published:August 27, 2013 at 2:05 PM Updated:August 30, 2024 at 2:35 PM Kolten Parker, Manager of Content and Coverage Downloading the BGC app in theApple StoreorGoogle Play Storeis the best way to get all Big Game Coverage from KSAT 12 sports — including watching highlights, reading insider analysis and more. This year, we've completely redesigned the app to make it easier to find the latest content. Recommended Videos This is a modal window. Begin...",
            "Published Datetime": "2013-08-27T19:05:33.000Z",
            "Published Text": "August 27, 2013 at 2:05 PM"
        },
        {
            "Category": "News",
            "URL": "https://www.ksat.com/news/local/2025/06/03/crash-closes-lanes-on-se-loop-410-on-south-side-drivers-urged-to-use-alternate-route-txdot-says/",
            "Headline": "Crash closes lanes on SE Loop 410 on South Side; drivers urged to use alternate route, TxDOT says",
            "Image": "https://res.cloudinary.com/...",
            "Description": "Ryan Cerna, Digital News Trainee Published:June 2, 2025 at 8:58 PM Ryan Cerna, Digital News Trainee SAN ANTONIO– Traffic is being diverted off Southeast Loop 410 at the Villamain Road exit due to a crash, TxDOT traffic cameras show. The crash was confirmed shortly before 7:45 p.m. at Loop 410 West at Moursund Boulevard, TxDOT said. All lanes are blocked and drivers are urged to use an alternate route. Recommended Videos This is a modal window. Beginning of dialog window. Escape will cancel and c...",
            "Published Datetime": "2025-06-03T01:58:02.333Z",
            "Published Text": "June 2, 2025 at 8:58 PM"
        },
        {
            "Category": "Sports",
            "URL": "https://www.ksat.com/sports/big-game-coverage/2025/04/25/brennans-javonte-johnson-commits-to-western-texas-college/",
            "Headline": "Brennan's Javonte Johnson commits to Western Texas College",
            "Image": "https://res.cloudinary.com/...",
            "Description": "Nick Mantas, Sports Editor Published:April 24, 2025 at 8:40 PM Nick Mantas, Sports Editor SAN ANTONIO– Brennan basketball star Javonte Johnson announced his commitment to Western Texas College on social media Wednesday. The 6-foot-5 guard helped Brennan reach the 6A state semifinal round this past season. Read more reporting and watch highlights and full games on theBig Game Coverage page. More Stories Like This In Our Email Newsletter Read also: Copyright 2025 by KSAT - All rights reserved. Nic...",
            "Published Datetime": "2025-04-25T01:40:33.105Z",
            "Published Text": "April 24, 2025 at 8:40 PM"
        },
        {
            "Category": "News",
            "URL": "https://www.ksat.com/station/2020/12/22/fcc-applications/",
            "Headline": "FCC Applications",
            "Image": "https://res.cloudinary.com/...",
            "Description": "Published:December 22, 2020 at 12:20 PM Updated:April 16, 2025 at 11:19 AM KSAT ATSC 1.0 Host Exhibit (for 3.0 Deployment) (Amended 2-22-24) (1)byJulie Morenoon Scribd KSAT Email Newsletters KSAT RSS Feeds Contests and Rules Contact Us KSAT Internships Careers at KSAT Closed Captioning / Audio Description Public File Current EEO Report Terms of Use Privacy Policy Do Not Sell My Info FCC Applications Cookie Preferences If you need help with the Public File, call(210) 351-1200 At KSAT, we are comm...",
            "Published Datetime": "2020-12-22T18:20:10.197Z",
            "Published Text": "December 22, 2020 at 12:20 PM"
        }
    ]

    for i, item in enumerate(sample_news_items):
        print(f"\n--- Analyzing Item {i+1} ---")
        print(f"Headline: {item.get('Headline')}")
        sentiment = analyze_news_item_sentiment(item)
        print(f"Predicted Sentiment: {sentiment}")
        