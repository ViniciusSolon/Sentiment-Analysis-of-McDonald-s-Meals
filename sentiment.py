import nltk
import streamlit as st
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the necessary lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Fictional data of comments about McDonald's meals
data = {
    'Comment': [
        "The Big Mac is wonderful, I love the taste!",
        "The McChicken was cold and tasteless.",
        "The McFlurry with Oreo is so good!",
        "The sandwich was too salty and the sauce was wrong.",
        "The fries were crispy and delicious!",
        "The Cheeseburger doesn't taste like meat, I didn't like it.",
        "The service was fast and the sandwich was perfect.",
        "The chocolate sundae is amazing, I love it!",
        "The McFish is the worst sandwich, I didn't like it.",
        "The packaging was all squashed and the sandwich was cold."
    ]
}

# Creating a DataFrame from the fictional data
df = pd.DataFrame(data)

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to classify the sentiment of comments
def classify_sentiment(comment):
    score = sid.polarity_scores(comment)
    if score["compound"] >= 0.05:
        return "Positive 😊"
    elif score["compound"] <= -0.05:
        return "Negative 😞"
    else:
        return "Neutral 😐"



# Start Streamlit interface
st.title('Sentiment Analysis: McDonald\'s Meals')
st.write("This project analyzes sentiment from comments about McDonald\'s meals.")

# Display the fictional data with sentiment classification
st.subheader('Comments and Sentiments')
st.dataframe(df)

# Display sentiment analysis for selected comment
selected_comment = st.selectbox(
    'Choose a comment for sentiment analysis',
    df['Comment']
)

# Display analysis for the selected comment
if selected_comment:
    sentiment = classify_sentiment(selected_comment)
    st.write(f"Comment: {selected_comment}")
    st.write(f"Sentiment: {sentiment}")


st.title("After the classification")

# Adding a sentiment column to the DataFrame
df['Sentiment'] = df['Comment'].apply(classify_sentiment)
st.dataframe(df)
