import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize NLTK resources for text preprocessing
nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained models and vectorizer for spam classification
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Create a function for text preprocessing
def transform_text(text):
    """
    Preprocesses the input text for classification.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        str: The preprocessed text.
    """
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove non-alphanumeric characters
    text = [i for i in text if i.isalnum()]
    # Remove stopwords and punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    # Stem the words
    text = [PorterStemmer().stem(i) for i in text]
    return " ".join(text)

# Streamlit UI
st.title("Email/SMS Spam Classifier")
st.markdown(
    '<style>h1{color: #007BFF;}</style>',
    unsafe_allow_html=True
)

# User input text area
input_text = st.text_area("Enter your message here", height=200, key='input_text')
input_text_placeholder = st.empty()  # Placeholder for input text

# Predict button with custom style and tooltip
predict_button = st.button('Predict', key='predict_button',
                           help="Click this button to classify the message as spam or not spam.")
if predict_button:
    if input_text:
        # Preprocess the input message
        transformed_text = transform_text(input_text)
        # Vectorize the input using TF-IDF
        vector_input = tfidf.transform([transformed_text])
        # Make a prediction using the trained model
        result = model.predict(vector_input)[0]
        # Display the prediction result with custom styling
        if result == 1:
            st.success("Prediction: Spam")
        else:
            st.success("Prediction: Not Spam")
    else:
        st.warning("Please enter a message before predicting.")
