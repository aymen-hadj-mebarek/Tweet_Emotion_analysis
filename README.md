# Sentiment Analysis with Neural Networks and LSTM

This project aims to classify text data as either positive or negative using a deep learning approach. The project employs both a simple Sequential Neural Network and an LSTM-based model to perform sentiment analysis. Text data is cleaned, preprocessed, and transformed into feature vectors before being fed into the models.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Setup](#project-setup)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Evaluation and Results](#evaluation-and-results)
- [File Descriptions](#file-descriptions)
- [Usage](#usage)
- [Future Work](#future-work)

---

### Project Overview

Sentiment analysis is the process of determining the sentiment or emotion expressed in a piece of text. This project builds two deep learning models:

1. A **Sequential Neural Network** for sentiment analysis using a simple feed-forward architecture.
2. An **LSTM Model** to capture sequential patterns in the text data, as sentiment often relies on the sequence of words.

Both models are trained and evaluated on a labeled dataset of text samples with corresponding sentiment labels.

---

### Project Setup

1. **Clone the Repository**:
    
    ```bash
    git clone https://github.com/aymen-hadj-mebarek/Tweet_Emotion_analysis.git
    ```
    
2. **Install Dependencies**: Install the required packages listed in `requirements.txt`:
    
    ```bash
    pip install -r requirements.txt
    ```
    
3. **Download NLTK Resources**:  
    The project requires additional NLTK data resources. Run the following commands in a Python shell:
    
    ```python
    import nltk nltk.download('stopwords') nltk.download('punkt') nltk.download('averaged_perceptron_tagger') nltk.download('wordnet')
    ```
    
4. **Dataset**: Place your dataset (`data.csv`) in the root directory. The dataset should contain two columns:
    
    - `target`: Sentiment label (e.g., 0 for negative, 1 for positive)
    - `text`: Text data containing the content for analysis.

---

### Data Preprocessing

The data preprocessing step involves cleaning and standardizing text data to improve model performance.

1. **Text Cleaning**:
    
    - Convert all text to lowercase for uniformity.
    - Remove URLs, mentions (e.g., `@username`), emojis, and special characters.
    - Replace common abbreviations or slang with full forms (e.g., "u" to "you").
2. **Stopword Removal**:
    
    - Remove commonly used words that do not contribute significant meaning, like "the," "is," "in," etc., using NLTK’s stopword list.
3. **Tokenization and Lemmatization**:
    
    - Tokenize the text into individual words for easier processing.
    - Apply lemmatization to reduce words to their root forms (e.g., "running" becomes "run"), ensuring consistency in word forms.

---

### Feature Engineering

1. **Vocabulary Creation**:
    - The vocabulary consists of unique words appearing in the dataset that meet a specified frequency threshold.
    - Each word in the vocabulary is assigned an index for vectorization.
2. **Binary Vectorization**:
    - Convert each text entry into a binary vector based on vocabulary presence (1 if the word is present, 0 if not).
    - This vectorization approach ensures the input features for the neural network are of fixed length.

---

### Model Training

1. **Train-Test Split**:
    
    - The dataset is split into training (80%) and testing (20%) sets to evaluate the model performance.
2. **Sequential Neural Network**:
    
    - The feed-forward neural network uses fully connected layers to predict sentiment.
    - Architecture:
        - **Input Layer**: Accepts binary vectors created from the vocabulary.
        - **Hidden Layers**: Two hidden layers with ReLU activation.
        - **Output Layer**: Single neuron with sigmoid activation for binary classification.
    - **Compilation and Training**:
        - The model is compiled with binary cross-entropy loss and the Adam optimizer.
        - It is trained over multiple epochs with early stopping to prevent overfitting.
3. **LSTM Model**:
    
    - The LSTM model uses sequences to capture order-based patterns in the data.
    - Data is reshaped to be compatible with LSTM input requirements.
    - Architecture:
        - **Embedding Layer**: Converts binary vectors into dense word embeddings.
        - **LSTM Layers**: Capture sequential dependencies in the text.
        - **Output Layer**: Single neuron with sigmoid activation.
    - **Compilation and Training**:
        - The model is trained with binary cross-entropy loss and Adam optimizer.
        - Early stopping is implemented based on validation loss.

---

### Evaluation and Results

1. **Metrics**:
    
    - Accuracy and loss are calculated for both models on test data.
    - Additional evaluation metrics such as Precision, Recall, and F1 Score can be included if needed.
2. **Visualization**:
    
    - **Class Distribution**: A bar plot displays the distribution of positive and negative samples in the dataset.
    - **Word Clouds**: Two word clouds are generated to visualize frequently occurring words in positive and negative samples.
3. **Comparison**:
    
    - Compare accuracy, training time, and generalization between the Sequential Neural Network and LSTM models.
    - Discuss any observed overfitting or underfitting and address model limitations.

---

### File Descriptions

- `data.csv`: Raw dataset file containing labeled text data.
- `clean_data.csv`: Processed dataset file saved for convenience, so the data cleaning steps do not need to be repeated.
- `vocabulary_and_frequencies.txt`: File containing the vocabulary with each word’s frequency count.
- `model_sequential.keras`: Saved model file for the sequential neural network.
- `LSTM_model.keras`: Saved model file for the LSTM network.
- `main.py`: Main script that orchestrates data preprocessing, model training, and evaluation.
- `requirements.txt`: List of Python libraries required to run the project.

---

### Usage

To run the project, execute the following command:

```bash
python main.py
```

Alternatively, open the Jupyter Notebook (`sentiment_analysis.ipynb`) to run and explore the project interactively.

---

### Future Work

Considerations for enhancing the project:

1. **Hyperparameter Tuning**: Experiment with different neural network architectures, learning rates, batch sizes, and regularization techniques to improve model performance.
2. **Additional Models**: Add other machine learning models, like Naive Bayes or SVM, for comparative analysis.
3. **Extended Preprocessing**: Implement more sophisticated preprocessing techniques, such as removing rare words, adding bigrams, or handling negations in text.
4. **Fine-tuning with Transformers**: Explore transformer models like BERT or RoBERTa for improved sentiment analysis performance.