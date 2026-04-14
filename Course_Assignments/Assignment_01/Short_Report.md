# Short Report: Amazon Reviews Preprocessing and Sentiment Analysis

## Overview
This report summarizes the preprocessing, feature engineering, and sentiment classification performed on the `Amazon_Reviews.csv` dataset. The dataset contains customer review records with fields including reviewer name, country, rating, review title, review text, and date of experience.

## Observations

### Data Preprocessing
- The review text was normalized by converting to lowercase, tokenizing, removing punctuation, filtering English stopwords, and applying lemmatization.
- Text preprocessing successfully handled missing or non-string values by replacing them with empty strings.
- This produced a clean `processed_text` column suitable for vectorization.

### Vocabulary Creation
- A vocabulary was built using `CountVectorizer` over the preprocessed review text.
- Vocabulary size: **24,902 unique terms**.
- Most frequent words in the dataset include:
  - `amazon`
  - `customer`
  - `service`
  - `item`
  - `delivery`
- Frequent terms reflect the dataset’s focus on service experience, delivery issues, and Amazon-specific feedback.

### Feature Engineering
- Three feature representations were generated:
  1. One Hot Encoding (binary presence of terms)
  2. Bag of Words (term frequency counts)
  3. TF-IDF (term frequency–inverse document frequency)
- All three sparse matrices have the same shape: **(21,214 documents, 24,902 features)**.

### Sparse Matrix Analysis
- Sparsity for OHE, BoW, and TF-IDF matrices: **99.87% zeros**.
- This level of sparsity is expected in text data because each document uses only a small fraction of the vocabulary.
- Sparse matrices are efficient for storage but require specialized operations for large-scale systems. Dense processing becomes impractical due to wasted memory and time.

## Comparison and Interpretation
- OHE captures only whether a word appears in a document.
- BoW encodes how frequently each word appears.
- TF-IDF weights each word by its importance, downweighting common words and upweighting words that are more unique to a document.
- In the sample review, TF-IDF assigned higher importance to distinctive terms such as `demanding`, `document`, `charging`, and `entered`.
- Common terms that appear across many reviews receive lower TF-IDF scores because they carry less discriminative power.

## Real-world Insights

### Why Bag of Words fails semantically
- BoW treats each word independently and ignores context, word order, and meaning relationships.
- Example: the words `bank` in `river bank` and `bank account` are represented identically in BoW.
- BoW also cannot capture negation or phrase-level meaning.

### When to use BoW and TF-IDF
- BoW is suitable for simple classification tasks, lightweight models, and cases where raw term frequency is informative.
- TF-IDF is more appropriate for search, information retrieval, document ranking, and tasks that benefit from term importance weighting.

### Limitations of TF-IDF
- TF-IDF still ignores word order and semantic meaning.
- It is sensitive to vocabulary coverage and struggles with out-of-vocabulary terms.
- It may require large corpora to estimate reliable IDF values.
- It does not generalize well to rich language phenomena or contextual embeddings.

## Sentiment Classification Use Case
- Sentiment labels were derived from the `Rating` field, mapping high ratings to positive sentiment and low ratings to negative sentiment.
- Two classification models were trained and evaluated:
  1. Logistic Regression
  2. Naive Bayes
- Both models were trained with BoW and TF-IDF features.

### Model Performance
- The dataset is imbalanced, with the positive class dominating.
- Reported accuracy is high (~99%), but this is misleading because the negative class is underrepresented.
- Both classifiers effectively predicted the majority positive class but failed to correctly classify the minority negative examples.
- The macro average F1-score is around **0.50**, indicating weak performance on the underrepresented class despite strong overall accuracy.

## Conclusions
- Preprocessing and feature engineering were successfully completed, producing robust BoW and TF-IDF representations.
- The dataset is highly sparse, which is typical for document-term matrices and underscores the need for sparse data handling.
- TF-IDF provides better term importance interpretation than raw frequency.
- The sentiment classification experiment highlights the danger of relying on accuracy alone when data are imbalanced.
- For real-world sentiment tasks, the next step should be to address imbalance with class weighting, resampling, or more advanced text embeddings.

## Recommendations
- Use TF-IDF rather than raw BoW for tasks requiring term discrimination and document ranking.
- Apply imbalance handling methods before training sentiment classifiers.
- Consider modern embedding-based approaches (e.g. word vectors or transformer embeddings) for better semantic understanding.
- Include a short evaluation of negative-class recall and F1 score in future experiments.
