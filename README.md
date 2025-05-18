# Tweets Sentiment Classification

This project focuses on classifying tweets based on their **sentiment** — positive, negative, or neutral — using **Natural Language Processing (NLP)** techniques and ensemble machine learning models.

## Project Overview

Social media platforms like Twitter are widely used to express opinions and emotions. This project leverages NLP to train models that can analyze the **sentiment** expressed in tweet text. The main goal is to accurately classify whether a tweet conveys a **positive**, **negative**, or **neutral** sentiment.


## Dataset Information

**Source:** [Sentiment and Emotions Labelled Tweets](https://www.kaggle.com/datasets/ankitkumar2635/sentiment-and-emotions-of-tweets) 

**Size:** ~25,000 manually labeled tweets  

**Fields:**
- `tweet_id`: Unique identifier of the tweet  
- `username`: Author of the tweet  
- `datetime`: Timestamp of the tweet  
- `text`: Raw tweet text  
- `sentiment`: Label indicating overall sentiment — **Positive**, **Negative**, or **Neutral**  
- `sentiment_score`: A float representing the intensity or confidence of the sentiment  
- `emotion`: *(not used in this notebook)*  
- `emotion_score`: *(not used in this notebook)*  

> 📌 This project uses **only the sentiment label** and ignores emotion-related columns.

## Technologies & Libraries

- **NLP & Embeddings**: `BERT Tokenizer`, `Transformers`, `re`, `string`  
- **Data Handling**: `pandas`, `numpy`  
- **Visualization**: `matplotlib`, `seaborn`  
- **Machine Learning**:
  - Base Models: `Logistic Regression`, `Random Forest`, `XGBoost`
  - Ensemble: `StackingClassifier` from Scikit-learn  
- **Evaluation**: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`

## Workflow Summary

1. **Text Preprocessing**:
   - Cleaned tweets (removing links, mentions, etc.)
   - Tokenization using **BERT**
   - Extracted embeddings using `BertModel`
2. **Label Encoding**: Converted sentiment labels to numeric
3. **Model Training**:
   - Trained several individual models and evaluated performance
   - Combined models using ensemble techniques (voting - stacking)
4. **Evaluation**: Used multiple metrics to assess model performance

## Final Results
- **Model**: Stacking Ensemble (`Logistic Regression` + `Random Forest` + `XGBoost`)
- **Accuracy**: 81.48 %
- **Precision**: 81.31 %
- **Recall**: 81.48 %
- **F1 Score**: 81.37 %

  ![normalized cm](https://github.com/user-attachments/assets/28d99d0a-6815-4ab0-99c4-1648233f02dc)


## Dashboard by [Roqia Adel](https://github.com/Roqia11)
An interactive Power BI Dashboard was developed by my teammate & freind [Roqia](https://github.com/Roqia11) to visualize tweet sentiments and emotions in an engaging, insightful way. It connects directly to the processed dataset and provides a comprehensive summary of sentiment analysis.

![Screenshot (428)](https://github.com/user-attachments/assets/d0aea940-c89d-423c-b44a-432448b80d49)

