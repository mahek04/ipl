#  IPL Performance Analysis Suite - README

This repository contains four comprehensive machine learning models for analyzing different aspects of Indian Premier League (IPL) cricket matches. Each model serves a distinct purpose and provides unique insights into match outcomes and player performances.

##  Table of Contents
1. [Match Winner Prediction](#1-match-winner-prediction)
2. [Toss Winner Classifier](#2-toss-winner-classifier)
3. [Batting Score Predictor](#3-batting-score-predictor)

---

## 1. Match Winner Prediction
** File:** `match_winner_prediction.py`  
** Objective:** Predict the winning team before the match begins using pre-match features  
** Techniques Used:**
- XGBoost Classifier
- One-Hot Encoding for categorical features
- Feature importance analysis

** Input Features:**
- Season year
- Venue
- Competing teams
- Toss winner and decision

** Output:**
- Predicted winning team
- Prediction probability
- Feature importance visualization

** Visualizations:**
1. Confusion matrix showing prediction accuracy
2. Feature importance chart (top 10 features)
3. Classification report with precision/recall metrics

** Key Insights:**
- Identifies which pre-match factors most influence outcomes
- Helps teams strategize based on venue and toss decisions
- Achieves ~70% accuracy in predicting winners pre-match

---

## 2. Toss Winner Classifier
** File:** `toss_winner_classifier.py`  
** Objective:** Predict which team will win the toss  
** Techniques Used:**
- Logistic Regression
- Feature engineering (temporal features from dates)
- Advanced model diagnostics

** Input Features:**
- Competing teams
- Venue
- Match date
- Historical toss patterns

** Output:**
- Probability of team1 winning toss
- Binary classification (team1/team2)

** Visualizations:**
1. Toss outcome distribution
2. ROC curve showing model discrimination
3. Calibration curve for probability reliability
4. Venue-specific toss trends

** Key Insights:**
- Some venues show strong toss bias
- Temporal patterns in toss outcomes
- ~65% accuracy in toss predictions

---

## 3. Batting Score Predictor
** File:** `batting_score_predictor.py`  
** Objective:** Predict final innings score during live matches  
** Techniques Used:**
- K-Nearest Neighbors Regression
- Over-by-over feature engineering
- Hyperparameter tuning with GridSearch

** Input Features:**
- Current over number
- Runs scored so far
- Wickets lost
- Run rate
- Batting team

** Output:**
- Predicted final score (range)
- Projected score confidence

** Visualizations:
1. Run rate progression through innings
2. Actual vs predicted scores scatter plot
3. Error distribution analysis
4. Over-wise prediction trends

** Key Insights:**
- Identifies key phases that impact final scores
- MAE of ~12 runs in predictions
- Powerplay performance strongly influences final totals


---

##  Getting Started
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Download datasets from Kaggle:
   - `matches.csv`
   - `deliveries.csv`
3. Run any model:
   ```bash
   python match_winner_prediction.py
   ```

##  Data Sources
All models use the [IPL Complete Dataset (2008-2020)](https://www.kaggle.com/patrickb1912/ipl-complete-dataset-20082020) from Kaggle

