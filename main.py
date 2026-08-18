import os
import re
import time
import nltk

from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# NLTK
# =========================================================

nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_text_from_pdf(file_path):
    """
    Extract text from all pages of a PDF.
    """

    reader = PdfReader(file_path)

    pages_text = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages_text.append(text)

    return "\n".join(pages_text)

# =========================================================
# TEXT PREPROCESSING
# =========================================================

def preprocess_text(text):
    """
    Clean and normalize text.
    """

    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Keep English alphabetic characters
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Tokenization
    words = text.split()

    # Stop Words Removal
    words = [
        word
        for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(words)

def split_into_sentences(text):
    """
    Split document text into sentences.
    """

    if not text:
        return []

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Basic sentence splitting
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return sentences

def find_similar_sentences(
    query,
    document_text,
    top_k=3
):
    """
    Find the most similar sentences
    between query and a document.
    """

    sentences = split_into_sentences(
        document_text
    )

    if not sentences:
        return []

    processed_query = preprocess_text(
        query
    )

    processed_sentences = [
        preprocess_text(sentence)
        for sentence in sentences
    ]

    # Remove empty sentences
    valid_pairs = []

    for original, processed in zip(
        sentences,
        processed_sentences
    ):

        if processed:

            valid_pairs.append(
                (original, processed)
            )

    if not valid_pairs:
        return []

    original_sentences = [
        pair[0]
        for pair in valid_pairs
    ]

    processed_sentences = [
        pair[1]
        for pair in valid_pairs
    ]

    # Create a temporary TF-IDF model
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        norm="l2"
    )

    sentence_matrix = (
        vectorizer.fit_transform(
            [processed_query]
            + processed_sentences
        )
    )

    query_vector = sentence_matrix[0]

    sentences_matrix = (
        sentence_matrix[1:]
    )

    similarities = cosine_similarity(
        query_vector,
        sentences_matrix
    )[0]

    ranked_indices = (
        similarities.argsort()[::-1]
    )

    results = []

    for index in ranked_indices[:top_k]:

        results.append({

            "sentence":
                original_sentences[index],

            "score":
                float(similarities[index])

        })

    return results
# =========================================================
# LOAD DOCUMENT COLLECTION
# =========================================================

def load_documents(folder_path):
    """
    Load PDF documents and keep both
    raw text and processed text.
    """

    documents = []
    document_names = []
    raw_documents = []

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for filename in sorted(os.listdir(folder_path)):

        if not filename.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(
            folder_path,
            filename
        )

        print(f"Loading: {filename}")

        try:

            raw_text = extract_text_from_pdf(
                file_path
            )

            processed_text = preprocess_text(
                raw_text
            )

            if not processed_text:

                print(
                    f"Skipped: {filename} "
                    f"(no extractable text)"
                )

                continue

            raw_documents.append(raw_text)

            documents.append(
                processed_text
            )

            document_names.append(
                filename
            )

        except Exception as error:

            print(
                f"Error loading {filename}: {error}"
            )

    return (
        documents,
        document_names,
        raw_documents
    )

# =========================================================
# TF-IDF MODEL
# =========================================================

def build_vectorizer(documents):
    """
    Build the TF-IDF representation
    for the document collection.
    """

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        norm="l2"
    )

    tfidf_matrix = vectorizer.fit_transform(
        documents
    )

    return vectorizer, tfidf_matrix

# =========================================================
# QUERY TEXT SEARCH
# =========================================================

def search_by_text(
    query,
    vectorizer,
    tfidf_matrix,
    document_names,
    raw_documents,
    top_k=3
):
    """
    Search documents using a text query
    and return similarity analysis.
    """

    processed_query = preprocess_text(
        query
    )

    if not processed_query:
        return []

    query_vector = vectorizer.transform(
        [processed_query]
    )

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    )[0]

    ranked_indices = similarities.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        common_terms = get_common_features(
            processed_query,
            index,
            vectorizer,
            tfidf_matrix
        )

        results.append({

            "document":
                document_names[index],

            "score":
                float(similarities[index]),

            "common_terms":
                common_terms,

            "common_count":
                len(common_terms),

            "document_index":
                int(index)
        })

    return results

# =========================================================
# QUERY DOCUMENT SEARCH
# =========================================================

def search_by_document(
    query_document_path,
    vectorizer,
    tfidf_matrix,
    document_names,
    raw_documents,
    top_k=3
):
    """
    Search using an external PDF document.
    """

    raw_query = extract_text_from_pdf(
        query_document_path
    )

    processed_text = preprocess_text(
        raw_query
    )

    if not processed_text:
        return []

    query_vector = vectorizer.transform(
        [processed_text]
    )

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    )[0]

    ranked_indices = (
        similarities.argsort()[::-1]
    )

    results = []

    for index in ranked_indices[:top_k]:

        common_terms = get_common_features(
            processed_text,
            index,
            vectorizer,
            tfidf_matrix
        )

        similar_sentences = (
            find_similar_sentences(
                raw_query,
                raw_documents[index],
                top_k=3
            )
        )

        results.append({

            "document":
                document_names[index],

            "score":
                float(similarities[index]),

            "common_terms":
                common_terms,

            "common_count":
                len(common_terms),

            "document_index":
                int(index),

            "similar_sentences":
                similar_sentences
        })

    return results

# =========================================================
# FEATURE ANALYSIS
# =========================================================

def get_query_features(
    query,
    vectorizer
):
    """
    Return the TF-IDF features
    that exist in the query.
    """

    query_vector = vectorizer.transform(
        [query]
    )

    feature_names = (
        vectorizer
        .get_feature_names_out()
    )

    row = query_vector.toarray()[0]

    features = []

    for index, value in enumerate(row):

        if value > 0:

            features.append({
                "term": feature_names[index],
                "weight": float(value)
            })

    features.sort(
        key=lambda x: x["weight"],
        reverse=True
    )

    return features

# =========================================================
# DOCUMENT FEATURE ANALYSIS
# =========================================================

def get_document_features(
    document_index,
    tfidf_matrix,
    vectorizer
):
    """
    Return TF-IDF features of
    a specific document.
    """

    feature_names = (
        vectorizer
        .get_feature_names_out()
    )

    row = tfidf_matrix[
        document_index
    ].toarray()[0]

    features = []

    for index, value in enumerate(row):

        if value > 0:

            features.append({
                "term": feature_names[index],
                "weight": float(value)
            })

    features.sort(
        key=lambda x: x["weight"],
        reverse=True
    )

    return features

# =========================================================
# COMMON FEATURES
# =========================================================

def get_common_features(
    query,
    document_index,
    vectorizer,
    tfidf_matrix
):
    """
    Find common TF-IDF features
    between query and document.
    """

    query_features = get_query_features(
        query,
        vectorizer
    )

    document_features = get_document_features(
        document_index,
        tfidf_matrix,
        vectorizer
    )

    query_terms = {
        item["term"]
        for item in query_features
    }

    document_terms = {
        item["term"]
        for item in document_features
    }

    common_terms = (
        query_terms &
        document_terms
    )

    return sorted(
        common_terms
    )

# =========================================================
# DISPLAY RESULTS
# =========================================================

def display_results(results):

    print()
    print("=" * 70)
    print("TOP 3 MOST SIMILAR DOCUMENTS")
    print("=" * 70)

    if not results:

        print("No results found.")

        return

    for rank, result in enumerate(
        results,
        start=1
    ):

        score = result["score"] * 100

        print()
        print(
            f"Rank #{rank}"
        )

        print(
            f"Document: "
            f"{result['document']}"
        )

        print(
            f"Similarity: "
            f"{score:.2f}%"
        )

        print(
            f"Common Features: "
            f"{result['common_count']}"
        )

        # -------------------------------------------------
        # Shared Terms
        # -------------------------------------------------

        if result["common_terms"]:

            print(
                "Shared Terms:"
            )

            print(
                ", ".join(
                    result["common_terms"][:15]
                )
            )

        else:

            print(
                "Shared Terms: None"
            )

        # -------------------------------------------------
        # Similar Sentences
        # -------------------------------------------------

        print()
        print(
            "Most Similar Sentences:"
        )

        similar_sentences = (
            result.get("similar_sentences",[])
        )

        if similar_sentences:

            for sentence_rank, item in enumerate(
                similar_sentences,
                start=1
            ):

                sentence_score = (
                    item["score"] * 100
                )

                print()
                print(
                    f"{sentence_rank}. "
                    f"{sentence_score:.2f}%"
                )

                print(
                    f"   {item['sentence']}"
                )

        else:

            print(
                "No similar sentences found."
            )

        print()
        print("-" * 70)

# =========================================================
# MAIN APPLICATION
# =========================================================

def main():

    documents_folder = "documents"

    print("=" * 70)
    print("DOCUMENT SIMILARITY SEARCH ENGINE")
    print("=" * 70)

    # -----------------------------------------------------
    # Load Corpus
    # -----------------------------------------------------

    start_time = time.time()

    documents, document_names, raw_documents = load_documents(
        documents_folder
    )

    loading_time = time.time() - start_time

    if not documents:

        print()
        print(
            "No valid PDF documents were found."
        )

        return

    print()
    print(
        f"Documents loaded: "
        f"{len(documents)}"
    )

    print(
        f"Loading time: "
        f"{loading_time:.3f} seconds"
    )

    # -----------------------------------------------------
    # Build Model
    # -----------------------------------------------------

    print()
    print("Building TF-IDF model...")

    model_start = time.time()

    vectorizer, tfidf_matrix = build_vectorizer(
        documents
    )

    model_time = time.time() - model_start

    print(
        f"TF-IDF matrix shape: "
        f"{tfidf_matrix.shape}"
    )

    print(
        f"Model building time: "
        f"{model_time:.3f} seconds"
    )

    # -----------------------------------------------------
    # Search Menu
    # -----------------------------------------------------

    while True:

        print()
        print("=" * 70)
        print("SEARCH MENU")
        print("=" * 70)

        print("1. Search using a sentence")
        print("2. Search using a PDF document")
        print("3. Exit")

        choice = input(
            "\nChoose an option: "
        ).strip()

        # -------------------------------------------------
        # Text Search
        # -------------------------------------------------

        if choice == "1":

            query = input(
                "\nEnter your sentence:\n> "
            ).strip()

            search_start = time.time()

            results = search_by_text(
                query,
                vectorizer,
                tfidf_matrix,
                document_names,
                raw_documents,
                top_k=3
            )

            search_time = (
                time.time() - search_start
            )

            display_results(results)

            print(
                f"Search time: "
                f"{search_time:.6f} seconds"
            )

        # -------------------------------------------------
        # PDF Search
        # -------------------------------------------------

        elif choice == "2":

            query_path = input(
                "\nEnter PDF path:\n> "
            ).strip()

            if not os.path.isfile(
                query_path
            ):

                print(
                    "File not found."
                )

                continue

            if not query_path.lower().endswith(
                ".pdf"
            ):

                print(
                    "Please provide a PDF file."
                )

                continue

            search_start = time.time()

            results = search_by_document(
                query_path,
                vectorizer,
                tfidf_matrix,
                document_names,
                raw_documents,
                top_k=3
            )

            search_time = (
                time.time() - search_start
            )

            display_results(results)

            print(
                f"Search time: "
                f"{search_time:.6f} seconds"
            )

        # -------------------------------------------------
        # Exit
        # -------------------------------------------------

        elif choice == "3":

            print(
                "\nExiting..."
            )

            break

        else:

            print(
                "Invalid choice."
            )

# =========================================================
# PROGRAM ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()