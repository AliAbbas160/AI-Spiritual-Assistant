import streamlit as st
from google import genai
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
import json
# ==============================
# CONFIG
# ==============================

import os

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

analyzer = SentimentIntensityAnalyzer()

bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# ==============================
# LOADING DATABASE
# ==============================

with open("Data/bhagavad_gita.json", "r", encoding="utf-8") as f:
    gita_data = json.load(f)

with open("Data/quran.json", "r", encoding="utf-8") as f:
    quran_data = json.load(f)

with open("Data/bible.json", "r", encoding="utf-8") as f:
    bible_data = json.load(f)

st.set_page_config(
    page_title="Historic Spiritual Assistant",
    page_icon="📜",
    layout="centered"
)

# ==============================
# HISTORIC THEME CSS
# ==============================

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background-color: #f4ecd8;
    background-image: radial-gradient(circle at 20% 20%, rgba(0,0,0,0.03) 1px, transparent 1px),
                      radial-gradient(circle at 80% 80%, rgba(0,0,0,0.03) 1px, transparent 1px);
    background-size: 50px 50px;
}

[data-testid="stHeader"] {
    background: transparent;
}

.title {
    text-align: center;
    font-size: 42px;
    font-family: "Georgia", serif;
    color: #000000;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    font-family: "Georgia", serif;
    color: #000000;
    margin-bottom: 30px;
}

.stSelectbox label, .stTextArea label {
    font-family: "Georgia", serif;
    color: #000000 !important;
    font-weight: bold;
}

.stButton>button {
    background-color: #000000;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-family: "Georgia", serif;
    border: none;
}

.stButton>button:hover {
    background-color: #333333;
}

.result-scroll {
    background-color: #fff8e7;
    padding: 25px;
    border-radius: 10px;
    margin-top: 20px;
    font-family: "Georgia", serif;
    color: black;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
}

.footer {
    text-align: center;
    color: black;
    margin-top: 30px;
    font-size: 13px;
    font-family: "Georgia", serif;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================

st.markdown('<div class="title">📜 AI Spiritual & Historical Manuscript</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Interpreting Sacred Texts through Ancient Wisdom & AI</div>', unsafe_allow_html=True)

# ==============================
# INPUT SECTION
# ==============================

religion = st.selectbox(
    "Select Sacred Scripture",
    ["Bhagavad Gita", "Quran", "Holy Bible"]
)

mode = st.selectbox(
    "Select Interpretation Style",
    ["Simple Explanation", "Spiritual Insight", "Historical Context"]
)

user_input = st.text_area(
    "Enter Verse or Question",
    placeholder="Example: Explain Bhagavad Gita 2.47",
    height=150
)

generate = st.button("📖 Reveal Interpretation")
compare = st.button("🔍 Compare Interpretations")

# ==============================
# SENTIMENT ANALYSIS
# ==============================

def analyze_sentiment(text):

    scores = analyzer.polarity_scores(text)

    compound = scores['compound']

    if compound >= 0.05:
        return "positive"

    elif compound <= -0.05:
        return "negative"

    else:
        return "neutral"

def load_scripture(religion):

    if religion == "Bhagavad Gita":
        return gita_data

    elif religion == "Quran":
        return quran_data

    else:
        return bible_data
# ==============================
# BERT SEMANTIC MATCHING
# ==============================



def get_similar_verse(user_query, religion):

    verses = load_scripture(religion)

    verse_texts = [v["text"] for v in verses]

    query_embedding = bert_model.encode(
        user_query,
        convert_to_tensor=True
    )

    verse_embeddings = bert_model.encode(
        verse_texts,
        convert_to_tensor=True
    )

    similarities = util.pytorch_cos_sim(
        query_embedding,
        verse_embeddings
    )

    best_index = similarities.argmax().item()

    return verses[best_index]

# ==============================
# PROMPT BUILDER
# ==============================


def build_prompt(religion, mode, user_input):

    sentiment = analyze_sentiment(user_input)
    similar_verse = get_similar_verse(
    user_input,
    religion
)

    # Tone based on sentiment
    if sentiment == "negative":
        tone = "Use a compassionate and reassuring tone."
    elif sentiment == "positive":
        tone = "Use an encouraging and positive tone."
    else:
        tone = "Use a neutral and balanced tone."

    base_prompt = f"""
You are an AI assistant specializing in interpreting {religion} scriptures.

Your task is to provide accurate, objective and easy-to-understand interpretations.

Instructions:
- Do NOT roleplay as a saint, guru, monk or ancient scholar.
- Do NOT greet the user.
- Do NOT use phrases like:
  • My dear seeker
  • Dear friend
  • Beloved soul
  • My child
  • O seeker
  • Brother or Sister
- Start directly with the explanation.
- Use modern, professional English.
- Keep the explanation factual and educational.
- Avoid unnecessary storytelling.

Detected Sentiment:
{sentiment}

Response Tone:
{tone}

Relevant Scripture

Reference:
{similar_verse['verse']}

Text:
{similar_verse['text']}

User Query:
{user_input}
"""

    # -----------------------------
    # SIMPLE EXPLANATION
    # -----------------------------
    if mode == "Simple Explanation":

        return base_prompt + """

Return ONLY in the following format.

## Plain Meaning
Explain the verse in simple language that anyone can understand.

## Key Takeaway
Provide 2-3 concise bullet points summarizing the main message.
"""

    # -----------------------------
    # SPIRITUAL INSIGHT
    # -----------------------------
    elif mode == "Spiritual Insight":

        return base_prompt + """

Return ONLY in the following format.

## Spiritual Insight
Explain the deeper philosophical and spiritual meaning.

## Practical Application
Explain how these teachings can be applied in daily life.

## Key Takeaway
Provide 2-3 concise bullet points summarizing the spiritual lesson.
"""

    # -----------------------------
    # HISTORICAL CONTEXT
    # -----------------------------
    elif mode == "Historical Context":

        return base_prompt + """

Return ONLY in the following format.

## Historical Background
Explain when and why this verse or teaching was revealed.

## Cultural Context
Describe the historical, cultural and religious significance.

## Historical Importance
Explain why this teaching is important within the scripture.

## Key Takeaway
Provide 2-3 concise bullet points summarizing the historical significance.
"""
# ==============================
# NORMAL GENERATION
# ==============================

if generate:
    if user_input.strip() == "":
        st.warning("Please enter a verse or question.")
    else:
        sentiment = analyze_sentiment(user_input)

        st.info(f"Detected Sentiment: {sentiment}")

        with st.spinner("Generating interpretation..."):
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=build_prompt(religion, mode, user_input)
            )

        formatted_text = response.text.replace("\n", "<br>")

        st.markdown(f"""
        <div class="result-scroll">
        <h3>🕯 Interpretation</h3>
        <p style="color:black; font-size:18px; line-height:1.6;">
        {formatted_text}
        </p>
        </div>
        """, unsafe_allow_html=True)

# ==============================
# COMPARE FEATURE
# ==============================

if compare:
    if user_input.strip() == "":
        st.warning("Please enter a verse or question.")
    else:
        with st.spinner("📖 Compare Interpretation Modes"):

            sentiment = analyze_sentiment(user_input)
            similar_verse = get_similar_verse(user_input,religion)

            spiritual_prompt = f"""
You are an AI assistant specializing in the interpretation of {religion} scriptures.

Detected Sentiment:
{sentiment}

Relevant Scripture

Reference:
{similar_verse['verse']}

Text:
{similar_verse['text']}

User Query:
{user_input}

Instructions:
- Do NOT roleplay as a guru, saint, monk, or ancient scholar.
- Do NOT greet the user.
- Do NOT use phrases such as:
  • My dear seeker
  • Dear friend
  • Beloved soul
  • My child
  • O seeker
  • Brother or Sister
- Start directly with the explanation.
- Use modern, professional English.
- Keep the response factual, respectful, and easy to understand.

Return the response in the following format:

## Spiritual Meaning
Explain the deeper spiritual message of the verse.

## Philosophical Insight
Discuss the underlying philosophy and life lessons.

## Practical Application
Explain how this teaching can be applied in everyday life.

## Key Takeaway
Provide 2-3 concise bullet points summarizing the message.
"""

            historical_prompt = f"""
You are an AI assistant specializing in the interpretation of {religion} scriptures.

Detected Sentiment:
{sentiment}

Relevant Scripture

Reference:
{similar_verse['verse']}

Text:
{similar_verse['text']}

User Query:
{user_input}

Instructions:
- Do NOT roleplay.
- Do NOT greet the user.
- Do NOT use phrases such as:
  • My dear seeker
  • Dear friend
  • Beloved soul
  • My child
  • O seeker
- Start directly with the explanation.
- Use clear and professional language.
- Focus only on historical and cultural aspects.

Return the response in the following format:

## Historical Background
Explain when and why this verse or teaching was delivered.

## Cultural Context
Describe the cultural, social, or religious significance.

## Historical Significance
Explain its importance in the scripture and its influence.

## Key Takeaway
Provide 2-3 concise bullet points summarizing the historical importance.
"""

            spiritual_res = client.models.generate_content(
                model="gemini-flash-latest",
                contents=spiritual_prompt
            )

            historical_res = client.models.generate_content(
                model="gemini-flash-latest",
                contents=historical_prompt
            )

        col1, col2 = st.columns(2)

        with col1:
            text1 = spiritual_res.text.replace("\n", "<br>")
            st.markdown(f"""
            <div class="result-scroll">
            <h3>🧘 Spiritual Insight</h3>
            <p style="color:black; font-size:16px; line-height:1.7;">
            {text1}
            </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            text2 = historical_res.text.replace("\n", "<br>")
            st.markdown(f"""
            <div class="result-scroll">
            <h3>📜 Historical Context</h3>
            <p style="color:black; font-size:16px; line-height:1.7;">
            {text2}
            </p>
            </div>
            """, unsafe_allow_html=True)
# ==============================
# FOOTER
# ==============================

st.markdown('<div class="footer">Disclaimer: AI-generated interpretation for educational purposes only.</div>', unsafe_allow_html=True)