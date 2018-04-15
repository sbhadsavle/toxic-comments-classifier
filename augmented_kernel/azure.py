
import requests
import pandas as pd
import numpy as np

import api_keys

subscription_key = api_keys.azure_text_api_key
text_analytics_base_url = "https://southcentralus.api.cognitive.microsoft.com/text/analytics/v2.0/"

language_api_url = text_analytics_base_url + "languages"
sentiment_api_url = text_analytics_base_url + "sentiment"
key_phrase_api_url = text_analytics_base_url + "keyPhrases"

def analyze_text(df_column):
    # Split df into chunks of less than 1000 comments each for API 
    chunks = np.array_split(df_column, 160)

    frames = []
    for i,df_chunk in enumerate(chunks):
        print("[Azure API] Analyzing chunk ", i)
        # Convert comments to documents
        documents = convert_to_documents(df_chunk)
        headers = {
            "Ocp-Apim-Subscription-Key": subscription_key
        }

        response = requests.post(language_api_url, headers=headers, json=documents)
        languages = response.json()
        # print(type(languages))

        response = requests.post(sentiment_api_url, headers=headers, json=documents)
        sentiments = response.json()

        response = requests.post(key_phrase_api_url, headers=headers, json=documents)
        key_phrases = response.json()
        # print(languages)
        # print(sentiments)
        # print(key_phrases)

        df = get_dataframe(languages, sentiments, key_phrases)
        frames.append(df)
        # print(df)

    result = pd.concat(frames, ignore_index=True)
    return result

def convert_to_documents(comments):
    """
    Converts a list of comments to a documents object that the Azure API
    expects.
    """
    documents = {
        "documents": []
    }
    id = 1
    for comment in comments:
        documents["documents"].append({
            "id": id,
            "text": comment
        })
        id += 1
    
    return documents

def extract_sentiments(documents):
    """Converts a documents object back to a list of sentiments.
    """
    sentiments = []
    for document in documents["documents"]:
        sentiments.append(document["score"])
    
    return sentiments
    

def get_dataframe(languages, sentiments, key_phrases):
    """Converts the text analytic results to a dataframe column.
    Args:
        languages: The languages json response.
        sentiments: The sentiments json response.
        key_phrases: The key phrases json response.
    """
    data = {
        "azure_sentiments": extract_sentiments(sentiments)
    }
    # print(data)
    return pd.DataFrame(data)

# data = {
#     "comment_text": [
#         "This is a comment",
#         "You are a nazi"
#     ]
# }
# df = pd.DataFrame(data)
# analyze_text(df["comment_text"])