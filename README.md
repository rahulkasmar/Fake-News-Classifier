# Fake News Classifier Using LSTM

This project is a machine learning-based Fake News Classifier that leverages a Long Short-Term Memory (LSTM) neural network to classify news articles as real or fake. It demonstrates the use of natural language processing (NLP) and deep learning techniques to tackle misinformation effectively.

## Features
- Preprocessing of text data to handle noise and inconsistencies.
- Tokenization and word embedding using techniques like GloVe or Word2Vec.
- LSTM-based neural network architecture for sequence modeling.
- Performance evaluation using metrics such as accuracy, precision, recall, and F1-score.

## Dataset
The project uses a publicly available dataset for fake news detection. If you wish to use the same dataset, you can download it from [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news/data) or a similar source.

## Installation
To run this project locally, ensure you have the following installed:

- Python 3.7+
- Jupyter Notebook or JupyterLab

Install the required Python libraries by running:
```bash
pip install -r requirements.txt
```

### Required Libraries
- pandas
- numpy
- matplotlib
- seaborn
- tensorflow
- scikit-learn
- nltk

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FakeNewsClassifierUsingLSTM.git
   ```
2. Navigate to the project directory:
   ```bash
   cd FakeNewsClassifierUsingLSTM
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook FakeNewsClassifierUsingLSTM.ipynb
   ```
4. Follow the steps in the notebook to preprocess the data, train the model, and evaluate its performance.

## Results
The LSTM model achieves:
- **Accuracy**: 90.37%

### Classification Report:
```
              precision    recall  f1-score   support

           0       0.91      0.92      0.92      3419
           1       0.89      0.89      0.89      2616

    accuracy                           0.90      6035
   macro avg       0.90      0.90      0.90      6035
weighted avg       0.90      0.90      0.90      6035
```

These metrics indicate the model's ability to distinguish between real and fake news articles effectively.

## Project Structure
```
FakeNewsClassifierUsingLSTM/
|-- FakeNewsClassifierUsingLSTM.ipynb  # Jupyter Notebook for the project
|-- requirements.txt                   # Python dependencies
|-- README.md                          # Project documentation
```

## Future Improvements
- Incorporate attention mechanisms to improve model performance.
- Use transformer-based models like BERT for better contextual understanding.
- Expand the dataset for broader generalization.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news/data)
- TensorFlow documentation
- Scikit-learn and NLTK libraries for preprocessing and evaluation.
