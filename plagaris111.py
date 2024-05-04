import streamlit as st
import nltk
import requests
from googlesearch import search as google_search
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import time
from googlesearch import search as google_search

nltk.download('punkt')
nltk.download('stopwords')

# Function to fetch Google search results with a delay between requests
def fetch_google_results(query, num_results=6, delay=2):
    results = []
    for result in google_search(query, num_results=num_results):
        results.append(result)
        time.sleep(delay)  # Introduce a delay between requests
    return results

# Function to preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Function to calculate Jaccard similarity
def jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# Function to calculate overlap coefficient
def overlap_coefficient(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = len(set1.intersection(set2))
    min_length = min(len(set1), len(set2))
    return intersection / min_length if min_length != 0 else 0

# Function to check plagiarism
def plagiarism_checker(query, document_text, threshold=0.3):
    plagiarized_links = []
    similarity_scores = []
    google_results = fetch_google_results(query)
    if google_results:
        for result in google_results:
            try:
                response = requests.get(result)
                soup = BeautifulSoup(response.text, 'html.parser')
                snippet_tag = soup.find('meta', attrs={'name': 'description'})
                if snippet_tag:
                    snippet = snippet_tag['content']
                    # Preprocess text for analysis
                    document_text_processed = preprocess_text(document_text)
                    snippet_processed = preprocess_text(snippet)

                    # Calculate similarity scores
                    vectorizer = TfidfVectorizer()
                    tfidf_matrix = vectorizer.fit_transform([document_text_processed, snippet_processed])
                    cosine_sim = cosine_similarity(tfidf_matrix)[0, 1]
                    jaccard_sim = jaccard_similarity(document_text_processed, snippet_processed)
                    overlap_coeff = overlap_coefficient(document_text_processed, snippet_processed)

                    # Check if any similarity score exceeds the threshold
                    if cosine_sim >= threshold or jaccard_sim >= threshold or overlap_coeff >= threshold:
                        plagiarized_links.append(result)
                        similarity_scores.append((cosine_sim, jaccard_sim, overlap_coeff))
            except Exception as e:
                pass
    
    return plagiarized_links, similarity_scores

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    text = ""
    pdf_reader = PyPDF2.PdfFileReader(file)
    for page_num in range(pdf_reader.numPages):
        text += pdf_reader.getPage(page_num).extractText()
    return text

# Streamlit app
def main():
    st.title("Plagiarism Checker")

    # Text input
    input_text = st.text_area("Enter the text to check for plagiarism:", height=200)

    # File upload
    uploaded_file = st.file_uploader("Upload a document file (PDF or TXT)", type=["pdf", "txt"])

    # Threshold input
    threshold = st.slider("Set the threshold for plagiarism detection:", min_value=0.1, max_value=1.0, value=0.3, step=0.1)

    if st.button("Check Plagiarism"):
        if uploaded_file is not None:
            # Extract text from uploaded file
            if uploaded_file.type == 'application/pdf':
                document_text = extract_text_from_pdf(uploaded_file)
            else:  # Assuming text file
                document_text = uploaded_file.getvalue().decode("utf-8")
                
            plagiarized_links, similarity_scores = plagiarism_checker(input_text, document_text, threshold)
            if plagiarized_links:
                st.write("Plagiarized Links:")
                for i, link in enumerate(plagiarized_links):
                    st.write(f"{i+1}. {link}")
                    st.write("Similarity Scores:")
                    st.write(f"- Cosine Similarity: {similarity_scores[i][0]}")
                    st.write(f"- Jaccard Similarity: {similarity_scores[i][1]}")
                    st.write(f"- Overlap Coefficient: {similarity_scores[i][2]}")
                    st.write("---")
            else:
                st.write("No plagiarized links found based on the threshold.")
        else:
            st.write("Please upload a document file (PDF or TXT).")

if __name__ == "__main__":
    main()
