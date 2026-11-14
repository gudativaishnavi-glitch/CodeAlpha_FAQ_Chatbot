import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Custom design
st.markdown("""
<style>
.main {
    background-color: #f4f7fb;
}
.title {
    text-align: center;
    color: #3157a4;
    font-size: 38px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #555;
    font-size: 17px;
}
</style>
""", unsafe_allow_html=True)

# FAQ database
faqs = {
    "What is artificial intelligence?":
    "AI is technology that enables machines to perform tasks that normally require human intelligence.",

    "What is machine learning?":
    "Machine Learning is a branch of AI that allows computers to learn from data.",

    "What is Python?":
    "Python is a popular programming language used in AI, web development and data science.",

    "What is NLP?":
    "NLP stands for Natural Language Processing. It helps computers understand human language.",

    "What is deep learning?":
    "Deep Learning is a branch of machine learning that uses neural networks.",

    "What is a chatbot?":
    "A chatbot is a computer program that communicates with users through text or voice.",

    "What is data science?":
    "Data Science involves collecting, analyzing and interpreting data.",

    "What is computer vision?":
    "Computer Vision helps computers understand images and videos.",

    "What is supervised learning?":
    "Supervised learning trains a model using labeled data.",

    "What is unsupervised learning?":
    "Unsupervised learning finds patterns in data without labeled answers."
}

questions = list(faqs.keys())
answers = list(faqs.values())

# NLP processing
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(questions)

# Find the most similar FAQ
def get_answer(user_question):
    user_vector = vectorizer.transform([user_question])
    similarity = cosine_similarity(user_vector, faq_vectors)

    best_match = similarity.argmax()
    score = similarity[0][best_match]

    if score < 0.20:
        return "Sorry, I don't know that answer yet. Please try another question!"

    return answers[best_match]

# Main heading
st.markdown(
    '<div class="title">🤖 AI FAQ Chatbot</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Your smart assistant for AI questions!</div>',
    unsafe_allow_html=True
)

st.write(" ")

# Sidebar
st.sidebar.title("⚙️ Chat Settings")
st.sidebar.write("Ask questions about Artificial Intelligence.")

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.subheader("Sample Questions")

for question in questions:
    st.sidebar.write("• " + question)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_input = st.chat_input("💬 Ask your question here...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    answer = get_answer(user_input)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()

st.markdown("---")
st.caption("AI FAQ Chatbot | Developed using Python, NLP and Streamlit")