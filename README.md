[README.md](https://github.com/user-attachments/files/22176383/README.md)
================================================================================\
                    AI FAKE REVIEW DETECTION SYSTEM\
                         ML Post-Bacc Capstone Project\
================================================================================\
\
PROJECT OVERVIEW\
----------------\
Machine learning system for detecting fake Amazon product reviews achieving \
91.39% F1 score. Combines text analysis with metadata features to identify \
fraudulent reviews with 96% precision.\
\
Author: Anoushka Ali\
Institution: Morehouse College\
Date: September 2024\
Target Performance: 85% F1 Score (Achieved: 91.39%)\
\
\
QUICK START\
-----------\
1. Install requirements:\
   pip install -r requirements.txt\
\
2. Run Jupyter notebook:\
   jupyter notebook AI_Fake_Reviews_Detection.ipynb\
\
3. For inference only:\
   python inference.py "Your review text here" 5\
\
\
FILES STRUCTURE\
---------------\
fake_review_detection/\
\uc0\u9474 \
\uc0\u9500 \u9472 \u9472  README.txt                          # This file\
\uc0\u9500 \u9472 \u9472  requirements.txt                     # Python dependencies\
\uc0\u9500 \u9472 \u9472  AI_Fake_Reviews_Detection.ipynb    # Main notebook\
\uc0\u9474 \
\uc0\u9500 \u9472 \u9472  data/\
\uc0\u9474    \u9500 \u9472 \u9472  fake reviews dataset.csv       # Dataset 1 (40,432 reviews)\
\uc0\u9474    \u9500 \u9472 \u9472  fake_reviews_dataset.csv       # Dataset 2 (40,432 reviews)\
\uc0\u9474    \u9492 \u9472 \u9472  final_labeled_fake_reviews.csv # Dataset 3 (50,000 reviews)\
\uc0\u9474 \
\uc0\u9500 \u9472 \u9472  models/\
\uc0\u9474    \u9500 \u9472 \u9472  Random_Forest_model.pkl        # Trained model (91.39% F1)\
\uc0\u9474    \u9500 \u9472 \u9472  tfidf_vectorizer.pkl          # Text vectorizer\
\uc0\u9474    \u9492 \u9472 \u9472  feature_scaler.pkl            # Feature scaler\
\uc0\u9474 \
\uc0\u9500 \u9472 \u9472  src/\
\uc0\u9474    \u9500 \u9472 \u9472  inference.py                   # Standalone prediction script\
\uc0\u9474    \u9492 \u9472 \u9472  app.py                         # Flask API (optional)\
\uc0\u9474 \
\uc0\u9492 \u9472 \u9472  results/\
    \uc0\u9500 \u9472 \u9472  confusion_matrix.png           # Model performance visuals\
    \uc0\u9500 \u9472 \u9472  roc_curves.png\
    \uc0\u9492 \u9472 \u9472  feature_importance.png\
\
\
DATASET INFORMATION\
-------------------\
Total Reviews: 130,852\
- Fake: 65,140 (49.78%)\
- Genuine: 65,712 (50.22%)\
\
Sources:\
- Kaggle Amazon review datasets\
- Updated within last 4 months\
- Balanced distribution for training\
\
\
MODEL PERFORMANCE\
-----------------\
Algorithm: Random Forest (100 estimators)\
\
Metrics:\
- F1 Score: 91.39% (Target: 85%)\
- Precision: 96.07%\
- Recall: 87.15%\
- Accuracy: 91.83%\
- ROC AUC: 97.08%\
\
Key Finding: Metadata features (word count, capitalization) more predictive \
than individual words.\
\
\
FEATURE ENGINEERING\
-------------------\
Total Features: 510\
\
Text Features (500):\
- TF-IDF vectorization with bigrams\
- Max 500 features\
\
Metadata Features (10):\
- word_count (8.65% importance)\
- char_count (8.25% importance)\
- avg_word_length (6.49% importance)\
- capital_ratio\
- sentiment_polarity\
- sentiment_subjectivity\
- all_caps_words\
- exclamation_count\
- rating\
- rating_deviation\
\
\
USAGE EXAMPLES\
--------------\
Python Script:\
    python inference.py "AMAZING PRODUCT!!!" 5\
    # Output: FAKE (89% confidence)\
\
Jupyter Notebook:\
    from joblib import load\
    model = load('models/Random_Forest_model.pkl')\
    # See notebook for full pipeline\
\
Flask API:\
    python app.py\
    # POST to http://localhost:5000/predict\
\
\
REPRODUCTION STEPS\
------------------\
1. Data Preparation:\
   - Load 3 CSV files\
   - Map labels: 'CG'\uc0\u8594 1 (fake), 'OR'\u8594 0 (genuine)\
   - Remove 12 rows with missing labels\
\
2. Feature Engineering:\
   - Extract metadata features\
   - Apply TF-IDF vectorization\
   - Scale metadata features\
\
3. Model Training:\
   - 80/20 train-test split\
   - Train Random Forest (100 trees)\
   - Evaluate on test set\
\
4. Save Models:\
   - Serialize with joblib\
   - Total size: ~52MB\
\
\
SYSTEM REQUIREMENTS\
-------------------\
- Python 3.8+\
- RAM: 8GB minimum (16GB recommended)\
- Storage: 500MB for models and data\
- OS: Windows/Mac/Linux\
\
\
KNOWN LIMITATIONS\
-----------------\
- English language only\
- Requires minimum 10 words per review\
- May not detect latest GPT-4 generated reviews\
- 13% false negative rate (misses some fakes)\
\
\
FUTURE IMPROVEMENTS\
-------------------\
- Multi-language support (Spanish, Mandarin)\
- GPT-4 detection features\
- Real-time learning pipeline\
- Image review analysis\
- Cross-platform adaptation (Yelp, TripAdvisor)\
\
\
TROUBLESHOOTING\
---------------\
Issue: Memory error during TF-IDF\
Solution: Reduce max_features from 500 to 200\
\
Issue: Slow training\
Solution: Use n_jobs=-1 for parallel processing\
\
Issue: Import errors\
Solution: Ensure all packages in requirements.txt installed\
\
\
CITATION\
--------\
If using this work, please cite:\
Ali, Anoushka. (2024). AI Fake Review Detection System. \
ML Post-Baccalaureate Capstone Project, Morehouse College.\
\
\
LICENSE\
-------\
MIT License - See LICENSE file for details\
\
\
CONTACT\
-------\
Author: A. Ali\
\
\
ACKNOWLEDGMENTS\
---------------\
- ML Post-Bacc Program Faculty\
- Kaggle dataset contributors\
- TACC computing resources (planned deployment)\
\
\
VERSION HISTORY\
---------------\
v1.0 - September 2025 - Initial release\
- Achieved 91.39% F1 score\
\
\
\
================================================================================\
                           END OF README\
================================================================================}
