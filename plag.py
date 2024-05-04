import streamlit as st
import nltk
import requests
from googlesearch import search
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textract
import docx

nltk.download('punkt')
nltk.download('stopwords')

# Function to fetch Google search results
def fetch_google_results(query, num_results=10):
    return list(search(query, num_results=num_results))

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
                        return "Plagiarism detected based on the threshold."
            except Exception as e:
                pass
    return "No plagiarism detected based on the threshold."

# Function to read text from PDF files
def read_pdf(file):
    text = textract.process(file)
    return text.decode('utf-8')

# Function to read text from DOCX files
def read_docx(file):
    doc = docx.Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Streamlit app
def main():
    st.title("Plagiarism Checker")

    # Text input
    option = st.radio("Choose Input Source:", ("Text Input", "File Upload"))

    if option == "Text Input":
        input_text = st.text_area("Enter the text to check for plagiarism:", height=200)
        threshold = st.slider("Set the threshold for plagiarism detection:", min_value=0.1, max_value=1.0, value=0.3, step=0.1)

        if st.button("Check Plagiarism"):
            result = plagiarism_checker(input_text, input_text, threshold)
            st.write(result)

    elif option == "File Upload":
        file1 = st.file_uploader("Upload the first file (Text, PDF, or DOCX)", type=["txt", "pdf", "docx"])
        file2 = st.file_uploader("Upload the second file (Text, PDF, or DOCX)", type=["txt", "pdf", "docx"])
        threshold = st.slider("Set the threshold for plagiarism detection:", min_value=0.1, max_value=1.0, value=0.3, step=0.1)

        if st.button("Check Similarity"):
            if file1 is not None and file2 is not None:
                if file1.type == "text/plain" and file2.type == "text/plain":
                    text1 = file1.getvalue().decode("utf-8")
                    text2 = file2.getvalue().decode("utf-8")
                    result = plagiarism_checker(text1, text2, threshold)
                    st.write(result)
                elif file1.type == "application/pdf" and file2.type == "application/pdf":
                    text1 = read_pdf(file1)
                    text2 = read_pdf(file2)
                    result = plagiarism_checker(text1, text2, threshold)
                    st.write(result)
                elif file1.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" and file2.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text1 = read_docx(file1)
                    text2 = read_docx(file2)
                    result = plagiarism_checker(text1, text2, threshold)
                    st.write(result)
                else:
                    st.write("Please upload files of the same type (Text, PDF, or DOCX) for comparison.")
            else:
                st.write("Please upload both files.")

if __name__ == "__main__":
    main()
