import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from zemberek import TurkishMorphology  # Turkish language processing library
import nltk
import string
import rarfile  # Library to handle .rar file extraction
from sklearn.neighbors import KNeighborsClassifier  # For k-NN classification
from sklearn.metrics.pairwise import cosine_similarity  # For calculating cosine similarity

nltk.download('punkt')  # For tokenizing text
nltk.download('stopwords')  # For stopwords removal

# Initialize Turkish morphological analyzer for stemming
morphology = TurkishMorphology.create_with_defaults()

# Function to extract .rar file contents
def extract_rar_file(rar_path, extract_to):
    with rarfile.RarFile(rar_path) as rf:
        rf.extractall(path=extract_to)

# Preprocesses text: lowercasing, deleting punctuation, discarding stop words, and stemming
def preprocess_text(text, stop_words):

    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    tokens = nltk.word_tokenize(text)  # Tokenize the text into words
    tokens = [token for token in tokens if token not in stop_words]  # Remove stop words

    # Stem tokens using TurkishMorphology
    stemmed_tokens = []
    for token in tokens:
        if token.isalnum():
            try:
                analysis = morphology.analyze_and_disambiguate(token).best_analysis()
                if analysis:
                    lemma = analysis[0].item.lemma  # Extract the lemma
                    stemmed_tokens.append(lemma)
            except Exception as e:
                print(f"Error processing token '{token}': {e}")

    return ' '.join(stemmed_tokens)

# Loads raw tweet data and their labels
def load_data(base_path):
    tweets = []
    labels = []
    categories = ["1", "2", "3"]  # Class labels as folder names

    for label, category in enumerate(categories):
        folder_path = os.path.join(base_path, "raw_texts", category)
        for file_name in os.listdir(folder_path):
            try:
                with open(os.path.join(folder_path, file_name), 'r', encoding='windows-1254') as file:
                    tweets.append(file.read().strip())
                    labels.append(label)  # Assign class label
            except UnicodeDecodeError:
                print(f"Error reading file {file_name}, skipping.")
    return tweets, labels

# Converts text data to TF-IDF representation
def get_tfidf_matrix(tweets, labels):
    vectorizer = TfidfVectorizer(max_features=1000)  # Limit to top 1000 features
    X = vectorizer.fit_transform(tweets)  # Transform tweets into a TF-IDF matrix

    # Convert the matrix to a DataFrame
    tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_df['Class'] = labels  # Add class labels to the DataFrame

    return tfidf_df, X, vectorizer

# Calculates performance metrics manually
def compute_metrics(y_true, y_pred, num_classes):
    metrics = {"precision": [], "recall": [], "f1_score": [],
               "true_positive": [], "false_positive": [], "false_negative": []}

    for cls in range(num_classes):
        true_positives = np.sum((y_true == cls) & (y_pred == cls))
        false_positives = np.sum((y_true != cls) & (y_pred == cls))
        false_negatives = np.sum((y_true == cls) & (y_pred != cls))

        # Calculate precision, recall, and F1 score
        precision = true_positives / (true_positives + false_positives) if (
            true_positives + false_positives) != 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (
            true_positives + false_negatives) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1_score"].append(f1)
        metrics["true_positive"].append(true_positives)
        metrics["false_positive"].append(false_positives)
        metrics["false_negative"].append(false_negatives)

    return metrics

# Performs stratified 10-fold cross-validation
def cross_validate(X, y, k_values):
    best_k = None
    best_f1 = 0
    best_metric = "cosine"
    results = []

    skf = StratifiedKFold(n_splits=10)  # 10-fold cross-validation

    for k in k_values:
        total_metrics = {"precision": [], "recall": [], "f1_score": [],
                         "true_positive": [], "false_positive": [], "false_negative": []}

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Compute similarity matrix for cosine similarity
            sim_matrix = cosine_similarity(X_train, X_test)

            # Train k-NN classifier
            model = KNeighborsClassifier(n_neighbors=k, metric='cosine')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = compute_metrics(y_test, y_pred, len(set(y)))  # Get metrics for this fold

            for key in total_metrics:
                total_metrics[key].append(metrics[key])

        # Calculate average metrics across all folds
        avg_metrics = {key: np.mean(total_metrics[key], axis=0) for key in total_metrics}
        macro_avg_f1 = np.mean(avg_metrics["f1_score"])

        # Update best k if current k yields better results
        if macro_avg_f1 > best_f1:
            best_f1 = macro_avg_f1
            best_k = k

        results.append((k, avg_metrics))

    return results, best_k, best_metric

def main():
    rar_path = #path to text file
    extract_to = os.path.join("extracted_data")
    extract_rar_file(rar_path, extract_to)

    # Load stop words in Turkish
    stop_words = nltk.corpus.stopwords.words('turkish')

    # Load tweets and their labels
    tweets, labels = load_data(extract_to)
    preprocessed_tweets = [preprocess_text(tweet, stop_words) for tweet in tweets]  # Preprocess tweets

    # Get the TF-IDF representation of the tweets
    tfidf_df, X, vectorizer = get_tfidf_matrix(preprocessed_tweets, labels)

    # Save TF-IDF results to CSV file
    tfidf_df.index = [f"Doc{i + 1}" for i in range(len(tweets))]
    tfidf_df.to_csv("tfidf_values.csv", index_label="Document")
    print("TF-IDF values saved to 'tfidf_values.csv'.")

    k_values = range(1, 10)  # Try different k values within the range

    # Uncomment one of these lines to try out different specific k values.
    # k_values = [3]
    # k_values = [5]

    # Perform cross-validation and compute performance metrics
    results, best_k, best_metric = cross_validate(X, np.array(labels), k_values)

    # Save the best performance metrics to a CSV file
    with open("best_performance_measures.csv", "w", newline='', encoding='utf-8') as f:
        f.write(f"Best results of k-NN obtained by: k={best_k}, similarity metric={best_metric}\n")

        headers = [
            "", "Class1", "Class2", "Class3", "MACRO Average", "Micro Average"
        ]
        f.write(", ".join(headers) + "\n")

        metrics_list = [
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F-Score", "f1_score"),
            ("Total no. of True Positive records", "true_positive"),
            ("Total no. of False Positive records", "false_positive"),
            ("Total no. of False Negative records", "false_negative")
        ]

        for metric_name, metric_key in metrics_list:
            row = [metric_name]

            # Calculate class-wise metrics
            if metric_key in ["true_positive", "false_positive", "false_negative"]:
                for class_idx in range(3):
                    row.append(np.sum([r[1][metric_key][class_idx] for r in results]))

                row.extend(["", ""])  # Leave Macro and Micro averages blank for counts
            else:
                for class_idx in range(3):
                    row.append(np.mean([r[1][metric_key][class_idx] for r in results]))

                # Calculate macro average
                macro_avg = np.mean([r[1][metric_key] for r in results], axis=0)
                row.append(np.mean(macro_avg))

                # Calculate micro average
                total_tp = np.sum([r[1]["true_positive"][0] for r in results])
                total_fp = np.sum([r[1]["false_positive"][0] for r in results])
                total_fn = np.sum([r[1]["false_negative"][0] for r in results])
                micro_avg = total_tp / (total_tp + total_fp + total_fn)
                row.append(micro_avg)

            f.write(", ".join(map(str, row)) + "\n")

    print("Best performance measures saved to 'best_performance_measures.csv'.")

if __name__ == '__main__':
    main()
