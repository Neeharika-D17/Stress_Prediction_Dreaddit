# Stress Prediction from Reddit Posts (Dreaddit Dataset)

This project builds a machine learning system to predict whether a Reddit post indicates **stress** or **no stress**.  
The Dreaddit dataset provides over 100 engineered linguistic, sentiment, and social features for each post, allowing models to learn stress patterns without using raw text.

## Dataset
- Files used: `dreaddit-train.csv`, `dreaddit-test.csv`
- Total samples: **3553**
- Features: **116 columns**
- Target label:
- `1` → Stressed
- `0` → Not stressed

## Preprocessing Steps
To prepare the data for modeling:

1. Removed the raw text column (`text`)
2. One-hot encoded:
   - `subreddit`
   - `post_id`
   - `sentence_range`
3. Scaled all numeric features for stability in training  
4. Final feature size after encoding: **3244 features**

## Models Trained
- Logistic Regression  
- Naive Bayes  
- Support Vector Machine  
- Decision Tree  
- AdaBoost Classifier  
- Neural Network (Dense model with dropout)

## Model Performance

| Model                        | Accuracy |
|------------------------------|----------|
| Logistic Regression          | 0.7356   |
| Naive Bayes                  | 0.5358   |
| SVM                          | 0.4767   |
| Decision Tree                | 0.6624   |
| **AdaBoost**                 | **0.7595** |
| Neural Network (scaled)      | 0.7384   |

**Best performer:** AdaBoost (~76%)

## How to Run the Project
1. Open the notebook: `Stress_Prediction_Dreaddit.ipynb`
2. Upload the dataset files:
   - `dreaddit-train.csv`
   - `dreaddit-test.csv`
3. Run all cells to train and evaluate the models.

## Requirements
Install the following Python libraries:
1. numpy
2. pandas
3. scikit-learn
4. tensorflow
5. matplotlib
6. seaborn
7. nltk

