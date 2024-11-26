import streamlit as st
import pandas as pd
from joblib import load

# Load model
knn_model = load(r"model_ex\knn_model.joblib")
book = load(r"model_ex\book.joblib")
book_pivot = load(r"model_ex\book_pivot.joblib")

# Set Streamlit page configuration
st.set_page_config(
    page_title="Bookie App",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS for better styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
            color: #212529;
        }
        h1 {
            color: #6f42c1;
        }
        .stButton button {
            background-color: #6f42c1;
            color: white;
            border: none;
            border-radius: 5px;
        }
        .stButton button:hover {
            background-color: white;
            color: #6f42c1
        }
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("📚 Book Recommender by Bookie App")
st.markdown("""
Welcome to **Bookie App**, your personal book recommender! 🎉  
Explore new books tailored to your tastes. Select a book and the number of recommendations you'd like, then let us do the rest!
""")

def recommend_book(name, count):
    books_lis, imgs_lis = [], []
    book_idx = book_pivot.index.get_loc(name)
    dist, idx = knn_model.kneighbors(
        book_pivot.iloc[book_idx, :].values.reshape(1, -1), n_neighbors=count + 1
    )  # Include self in neighbor
    for i in range(1, len(idx[0])):  # Skip self (index 0)
        idx_book = idx[0][i]
        books_lis.append(book_pivot.index[int(idx_book)])
        imgs_lis.append(book[book['Title'] == book_pivot.index[int(idx_book)]]['Image'].values[0])
    return books_lis, imgs_lis

# Sidebar for input
with st.sidebar:
    st.header("📖 Choose your preferences")
    selected_book = st.selectbox("Select a Book", book_pivot.index)
    count = st.slider(
        "Number of Recommendations", min_value=1, max_value=20, value=5
    )

# Show recommendations
if st.button('💡 Show Recommendations'):
    st.subheader("✨ Here are your recommended books:")
    text, img_url = recommend_book(selected_book, count)

    # Display recommendations in rows and columns
    max_cols = 5  # Max books per row
    rows = (len(text) + max_cols - 1) // max_cols  # Calculate the number of rows
    idx = 0  # Initialize book index

    for _ in range(rows):
        cols = st.columns(max_cols)
        for col in cols:
            if idx < len(text):  # Check if there are more books to display
                with col:
                    st.image(img_url[idx], caption=text[idx])
                    idx += 1
            else:
                break

# Footer
st.markdown("""
---
**🌟 Explore new worlds, one book at a time!**  
💡 *Built with Streamlit* | 🚀 *Your predictive companion*
""")
