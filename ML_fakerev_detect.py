# %%
# AI Fake Reviews Detection Model
# Capstone Project 

# ============================================
# PART 1: SETUP AND IMPORTS
# ============================================

# First, install the missing packages
!pip install textblob
!pip install shap  # Added installation for the missing shap package
!pip install lime  # Added installation for the missing lime package

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob  # This will work after installing the package

# Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Model Building
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve

# For interpretability
import shap  # This will work after installing the package
from lime.lime_text import LimeTextExplainer  # This will work after installing lime

# Download required NLTK data
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

print("All libraries imported successfully!")

# %%
# ============================================
# PART 2: DATA LOADING AND COMBINING
# ============================================

def load_and_combine_datasets(file_paths):
    """
    fake reviews dataset.csv,
    fake_reviews_dataset.csv,
    final_labeled_fake_reviews.csv

    """
    datasets = []
    
    for idx, path in enumerate(file_paths):
        try:
            # Add error_bad_lines=False and warn_bad_lines=True to skip problematic rows
            # Use dtype=str to prevent automatic type conversion issues with NaN values
            df = pd.read_csv(path, dtype=str)  # Read all columns as strings initially
            df['source_dataset'] = f'dataset_{idx+1}'  # Track which dataset each review came from
            datasets.append(df)
            print(f"Dataset {idx+1} loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"Columns: {df.columns.tolist()}\n")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    # Combine all datasets
    combined_df = pd.concat(datasets, ignore_index=True)
    print(f"\nCombined dataset shape: {combined_df.shape}")
    
    return combined_df

# Make sure pandas is imported
import pandas as pd

# PART 2: Reload the data fresh
file_paths = [
    'fake reviews dataset.csv',
    'fake_reviews_dataset.csv',
    'final_labeled_fake_reviews.csv'
]

# Load and combine
df = load_and_combine_datasets(file_paths)
print(f"Initial combined shape: {df.shape}")

# Define standardize_columns function if it's not defined elsewhere
def standardize_columns(df):
    # Keep all columns as strings to avoid NaN conversion issues
    # You can add specific column conversions here as needed
    # For example:
    # if 'rating' in df.columns:
    #     df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    # Return the dataframe with standardized columns
    return df

# Standardize columns
df = standardize_columns(df)

print(f"Using combined dataset with {len(df)} reviews")
print("Columns:", df.columns.tolist())
print(df.head())

# %%
# ============================================
# PART 3: Fixed standardization function
# ============================================

def standardize_columns_fixed(df):
    """
    Properly standardize columns for the three datasets
    """
    print(f"Starting with {len(df)} rows")
    
    # First, let's see what we're working with
    print("Initial columns:", df.columns.tolist())
    print("Initial label unique values:", df['label'].unique() if 'label' in df.columns else "No label column")
    
    # Create review_text from text_ or text
    if 'text_' in df.columns and 'text' in df.columns:
        df['review_text'] = df['text'].fillna(df['text_'])
        df = df.drop(['text_', 'text'], axis=1)
    elif 'text_' in df.columns:
        df['review_text'] = df['text_']
        df = df.drop(['text_'], axis=1)
    elif 'text' in df.columns:
        df['review_text'] = df['text']
        df = df.drop(['text'], axis=1)
    
    # Rename other columns
    rename_map = {
        'helpful_vote': 'helpful_votes',
    }
    df = df.rename(columns=rename_map)
    
    # Remove rows with missing review_text
    before_drop = len(df)
    df = df.dropna(subset=['review_text'])
    print(f"Dropped {before_drop - len(df)} rows with missing review text")
    
    # Fix labels - IMPORTANT: handle all cases
    if 'label' in df.columns:
        # Convert everything to string first to handle mixed types
        df['label'] = df['label'].astype(str)
        
        label_mapping = {
            'CG': 1,    # Computer Generated = Fake
            'OR': 0,    # Original = Genuine  
            '1': 1,
            '0': 0,
            '1.0': 1,
            '0.0': 0,
        }
        
        df['label'] = df['label'].map(label_mapping)
        
        # Check unmapped
        unmapped = df['label'].isna().sum()
        if unmapped > 0:
            print(f"Removing {unmapped} rows with unmapped labels")
            print("Unmapped label values were:", df[df['label'].isna()]['label'].unique())
            df = df.dropna(subset=['label'])
        
        df['label'] = df['label'].astype(int)
    
    print(f"Final shape: {df.shape}")
    print(f"Final columns: {df.columns.tolist()}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print(f"Percentage fake: {(df['label']==1).mean()*100:.2f}%")
    
    return df

# Apply the fixed standardization
df = standardize_columns_fixed(df)

# %%
df = standardize_columns(df)
print("\nColumns after standardization:", df.columns.tolist())

# %%
# Check what columns we have
print("Current columns:", df.columns.tolist())

# Fix the review_text column based on what exists
if 'review_text' not in df.columns:
    if 'text' in df.columns and 'text_' in df.columns:
        print("Combining 'text' and 'text_' columns...")
        df['review_text'] = df['text'].fillna(df['text_'])
    elif 'text_' in df.columns:
        print("Using 'text_' column...")
        df['review_text'] = df['text_']
    elif 'text' in df.columns:
        print("Using 'text' column...")
        df['review_text'] = df['text']
    else:
        print("ERROR: No text column found!")
        print("Available columns:", df.columns.tolist())
else:
    print("review_text column already exists")

# Only proceed if review_text was created
if 'review_text' in df.columns:
    # Remove rows with no review text
    initial_shape = df.shape
    df = df.dropna(subset=['review_text'])
    print(f"Removed {initial_shape[0] - df.shape[0]} rows with missing reviews")
    
    # Fix the labels
    print("\nFixing labels...")
    print(f"Unique labels before mapping: {df['label'].unique()}")
    
    label_mapping = {
        'CG': 1,  # Computer Generated = Fake
        'OR': 0,  # Original = Genuine  
        '1': 1,
        '0': 0,
        1: 1,
        0: 0
    }
    
    df['label'] = df['label'].map(label_mapping)
    
    # Check for unmapped labels
    unmapped = df['label'].isna().sum()
    if unmapped > 0:
        print(f"Warning: {unmapped} labels couldn't be mapped")
        df = df.dropna(subset=['label'])
    
    df['label'] = df['label'].astype(int)
    
    print(f"\n=== FINAL DATASET ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"\nPercentage fake reviews: {(df['label']==1).mean()*100:.2f}%")
    print(f"\nSample of data:")
    print(df[['review_text', 'label', 'rating']].head())
else:
    print("ERROR: Could not create review_text column")

# %%
# ============================================
# PART 4: EXPLORATORY DATA ANALYSIS
# ============================================

def perform_eda(df):
    """
    Perform exploratory data analysis
    """
    print("\n=== DATASET OVERVIEW ===")
    print(f"Total reviews: {len(df)}")
    print(f"Features: {df.columns.tolist()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    if 'label' in df.columns:
        print(f"\nClass distribution:")
        print(df['label'].value_counts())
        print(f"Percentage of fake reviews: {(df['label']==1).mean()*100:.2f}%")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Class distribution
    if 'label' in df.columns:
        df['label'].value_counts().plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Distribution of Fake vs Genuine Reviews')
        axes[0,0].set_xticklabels(['Genuine', 'Fake'], rotation=0)
    
    # 2. Rating distribution
    if 'rating' in df.columns:
        df['rating'].value_counts().sort_index().plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Rating Distribution')
    
    # 3. Review length distribution
    if 'review_length' in df.columns:
        axes[1,0].hist(df['review_length'], bins=30, edgecolor='black')
        axes[1,0].set_title('Review Length Distribution')
        axes[1,0].set_xlabel('Review Length')
    
    # 4. Verified purchase distribution
    if 'verified_purchase' in df.columns:
        df['verified_purchase'].value_counts().plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Verified Purchase Distribution')
    
    plt.tight_layout()
    plt.show()
    
    return df


perform_eda(df)

# %%
# PART 4.5: DATA CLEANING - Focus on Core Features Only

print("=== CLEANING DATA FOR MODELING ===")
print(f"Starting shape: {df.shape}")

# Check missing percentages for all columns
print("\nMissing value percentages:")
for col in df.columns:
    missing_pct = df[col].isna().sum() / len(df) * 100
    if missing_pct > 0:
        print(f"  {col}: {missing_pct:.1f}% missing")

# Keep only essential columns with complete/mostly complete data
essential_columns = [
    'review_text',  # Main feature (required) - 0% missing
    'rating',       # Important feature - 0% missing  
    'label'         # Target variable (required) - 0% missing
]

print(f"\nKeeping only essential columns: {essential_columns}")
df_clean = df[essential_columns].copy()

# Verify no missing values in these columns
print(f"\nMissing values in essential columns:")
print(df_clean.isnull().sum())

# Just to be safe, remove any rows with nulls
initial_len = len(df_clean)
df_clean = df_clean.dropna()
if initial_len != len(df_clean):
    print(f"Removed {initial_len - len(df_clean)} rows with missing values")

print(f"\n=== FINAL CLEANED DATA ===")
print(f"Shape: {df_clean.shape}")
print(f"Columns: {df_clean.columns.tolist()}")
print(f"Confirmed no missing values: {df_clean.isnull().sum().sum() == 0}")
print(f"\nLabel distribution:")
print(df_clean['label'].value_counts())
print(f"Percentage fake: {(df_clean['label']==1).mean()*100:.2f}%")

# Replace df with the cleaned version
df = df_clean
print("\n✅ Data cleaning complete! Ready for feature engineering with core features.")

# %%
# FIX: Convert rating to numeric
print("Fixing rating column...")
print(f"Rating unique values before fix: {df['rating'].unique()[:20]}")  # Show first 20 unique values

# Convert rating to numeric, handling any non-numeric values
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Check for any NaN values created by conversion
nan_ratings = df['rating'].isna().sum()
if nan_ratings > 0:
    print(f"Found {nan_ratings} non-numeric ratings, removing them...")
    df = df.dropna(subset=['rating'])

# Ensure rating is in valid range (1-5)
df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]

print(f"Rating unique values after fix: {sorted(df['rating'].unique())}")
print(f"Rating data type: {df['rating'].dtype}")
print(f"Final shape: {df.shape}")

# %%
# ============================================
# PART 5: FEATURE ENGINEERING
# ============================================

def extract_text_features(text):
    """
    Extract various text-based features from review
    """
    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text)
    
    features = {}
    
    # Basic features
    features['char_count'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
    
    # Punctuation features
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['punctuation_count'] = sum([1 for char in text if char in '!?.,;:'])
    
    # Capitalization features
    features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    features['all_caps_words'] = sum(1 for word in text.split() if word.isupper())
    
    # Sentiment features (using TextBlob)
    try:
        blob = TextBlob(text)
        features['sentiment_polarity'] = blob.sentiment.polarity
        features['sentiment_subjectivity'] = blob.sentiment.subjectivity
    except:
        features['sentiment_polarity'] = 0
        features['sentiment_subjectivity'] = 0
    
    return features

def create_features(df):
    """
    Create all features for the model
    """
    print("Extracting text features...")
    
    # Apply text feature extraction
    text_features = df['review_text'].apply(extract_text_features)
    text_features_df = pd.DataFrame(list(text_features))
    
    # Combine with original dataframe - use reset_index to avoid issues
    df = pd.concat([df.reset_index(drop=True), text_features_df.reset_index(drop=True)], axis=1)
    
    # Add rating-based features now that rating is numeric
    if 'rating' in df.columns:
        df['rating_deviation'] = abs(df['rating'] - df['rating'].mean())
        df['rating_extremity'] = abs(df['rating'] - 3)
    
    print(f"Total features created: {len(df.columns)}")
    print(f"New columns added: {text_features_df.columns.tolist() + ['rating_deviation', 'rating_extremity']}")
    
    return df

# Apply feature creation
df = create_features(df)
print("\nFeatures after engineering:", df.columns.tolist())
print("Shape after feature engineering:", df.shape)
print("\nData types after feature engineering:")
print(df.dtypes)


# %%
# ============================================
# PART 6: TEXT PREPROCESSING
# ============================================

def preprocess_text(text):
    """
    Clean and preprocess text data
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Apply preprocessing
df['processed_text'] = df['review_text'].apply(preprocess_text)
print("Text preprocessing completed")

# %%
# ============================================
# PART 7: FEATURE VECTORIZATION
# ============================================

def prepare_features(df, max_features=1000):
    """
    Prepare final feature matrix combining text and metadata
    """
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    tfidf_features = tfidf.fit_transform(df['processed_text'])
    tfidf_df = pd.DataFrame(tfidf_features.toarray(), 
                            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
    
    # Select metadata features
    metadata_features = ['char_count', 'word_count', 'avg_word_length', 
                         'exclamation_count', 'capital_ratio', 'all_caps_words',
                         'sentiment_polarity', 'sentiment_subjectivity']
    
    # Add available metadata features
    available_metadata = [col for col in metadata_features if col in df.columns]
    if 'rating' in df.columns:
        available_metadata.append('rating')
    if 'verified_purchase' in df.columns:
        available_metadata.append('verified_purchase')
    if 'rating_deviation' in df.columns:
        available_metadata.append('rating_deviation')
    
    metadata_df = df[available_metadata].fillna(0)
    
    # Standardize metadata features
    scaler = StandardScaler()
    metadata_scaled = scaler.fit_transform(metadata_df)
    metadata_scaled_df = pd.DataFrame(metadata_scaled, columns=available_metadata)
    
    # Combine all features
    X = pd.concat([tfidf_df, metadata_scaled_df], axis=1)
    y = df['label']
    
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Features include: {X.shape[1]} TF-IDF features + {len(available_metadata)} metadata features")
    
    return X, y, tfidf, scaler

X, y, tfidf_vectorizer, scaler = prepare_features(df, max_features=500)



# %%
# ============================================
# PART 8: MODEL TRAINING & EVALUATION
# ============================================

import time
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MinMaxScaler

# Split the data (if not already done)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models and compare performance
    """
    results = {}
    
    # 1. LOGISTIC REGRESSION
    print(f"\nTraining Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
    
    results['Logistic Regression'] = {
        'model': lr_model,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'predictions': y_pred
    }
    
    print(f"Accuracy: {results['Logistic Regression']['accuracy']:.4f}")
    print(f"Precision: {results['Logistic Regression']['precision']:.4f}")
    print(f"Recall: {results['Logistic Regression']['recall']:.4f}")
    print(f"F1 Score: {results['Logistic Regression']['f1_score']:.4f}")
    print(f"ROC AUC: {results['Logistic Regression']['roc_auc']:.4f}")
    
    # 2. RANDOM FOREST
    print(f"\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    results['Random Forest'] = {
        'model': rf_model,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'predictions': y_pred
    }
    
    print(f"Accuracy: {results['Random Forest']['accuracy']:.4f}")
    print(f"Precision: {results['Random Forest']['precision']:.4f}")
    print(f"Recall: {results['Random Forest']['recall']:.4f}")
    print(f"F1 Score: {results['Random Forest']['f1_score']:.4f}")
    print(f"ROC AUC: {results['Random Forest']['roc_auc']:.4f}")
    
    # 3. SVM (LinearSVC for speed)
    print(f"\nTraining SVM (LinearSVC for speed)...")
    start_time = time.time()
    
    svm_model = LinearSVC(random_state=42, max_iter=1000)
    svm_model.fit(X_train, y_train)
    
    # Calibrate for probability scores
    calibrated_svm = CalibratedClassifierCV(svm_model, cv='prefit')
    calibrated_svm.fit(X_train, y_train)
    y_pred = calibrated_svm.predict(X_test)
    y_pred_proba = calibrated_svm.predict_proba(X_test)[:, 1]
    
    results['SVM'] = {
        'model': calibrated_svm,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'predictions': y_pred
    }
    
    print(f"Accuracy: {results['SVM']['accuracy']:.4f}")
    print(f"Precision: {results['SVM']['precision']:.4f}")
    print(f"Recall: {results['SVM']['recall']:.4f}")
    print(f"F1 Score: {results['SVM']['f1_score']:.4f}")
    print(f"ROC AUC: {results['SVM']['roc_auc']:.4f}")
    print(f"Training time: {time.time() - start_time:.2f} seconds")
    
    # 4. NAIVE BAYES
    print(f"\nTraining Naive Bayes...")
    
    # MinMaxScaler for non-negative values
    scaler_nb = MinMaxScaler()
    X_train_nb = scaler_nb.fit_transform(X_train)
    X_test_nb = scaler_nb.transform(X_test)
    
    nb_model = MultinomialNB()
    nb_model.fit(X_train_nb, y_train)
    y_pred = nb_model.predict(X_test_nb)
    y_pred_proba = nb_model.predict_proba(X_test_nb)[:, 1]
    
    results['Naive Bayes'] = {
        'model': nb_model,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'predictions': y_pred
    }
    
    print(f"Accuracy: {results['Naive Bayes']['accuracy']:.4f}")
    print(f"Precision: {results['Naive Bayes']['precision']:.4f}")
    print(f"Recall: {results['Naive Bayes']['recall']:.4f}")
    print(f"F1 Score: {results['Naive Bayes']['f1_score']:.4f}")
    print(f"ROC AUC: {results['Naive Bayes']['roc_auc']:.4f}")
    
    return results

# Train all models
results = train_models(X_train, X_test, y_train, y_test)
print("\n✅ All models trained successfully!")

# %%
# ============================================
# PART 9: MODEL COMPARISON & VISUALIZATION
# ============================================

def visualize_results(results, y_test):
    """
    Visualize model comparison
    """
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        name: {
            'Accuracy': res['accuracy'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'F1 Score': res['f1_score'],
            'ROC AUC': res['roc_auc']
        }
        for name, res in results.items()
    }).T
    
    print("\n=== MODEL COMPARISON ===")
    print(comparison_df.round(4))
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Metrics comparison
    comparison_df.plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Model Performance Comparison')
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].set_ylim([0, 1])
    
    # 2. F1 Score comparison
    f1_scores = [res['f1_score'] for res in results.values()]
    axes[0, 1].bar(results.keys(), f1_scores, color=['blue', 'green', 'red', 'orange'])
    axes[0, 1].set_title('F1 Score Comparison')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].axhline(y=0.85, color='r', linestyle='--', label='Target (85%)')
    axes[0, 1].legend()
    
    # 3. Confusion Matrix for best model
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    best_model = results[best_model_name]
    cm = confusion_matrix(y_test, best_model['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # 4. ROC Curves
    for name, res in results.items():
        model = res['model']
        if name == 'Naive Bayes':
            from sklearn.preprocessing import MinMaxScaler
            scaler_nb = MinMaxScaler()
            X_test_nb = scaler_nb.fit_transform(X_test)
            y_pred_proba = model.predict_proba(X_test_nb)[:, 1]
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        axes[1, 1].plot(fpr, tpr, label=f"{name} (AUC: {res['roc_auc']:.3f})")
    
    axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curves')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return best_model_name, comparison_df

best_model_name, comparison_df = visualize_results(results, y_test)
print(f"\nBest performing model: {best_model_name}")

# %%
# ============================================
# PART 10: Quick HYPERPARAMETER TUNING
# ============================================

# Quick hyperparameter tuning - smaller grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,  # Reduced from 5
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best F1: {grid_search.best_score_:.4f}")


# %%
# ==================================================
# PART 11: - Feature Importance from Random Forest
# ==================================================

print("\n=== MODEL INTERPRETABILITY ===")

# Since Random Forest is your best model
best_model = results['Random Forest']['model']

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

# Show top 20 features
top_features = feature_importance.head(20)
print("\nTop 20 Most Important Features:")
print(top_features)

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(top_features['feature'][:15], top_features['importance'][:15])
plt.xlabel('Importance')
plt.title('Top 15 Most Important Features for Fake Review Detection')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Identify which are TF-IDF vs metadata features
tfidf_features = top_features[top_features['feature'].str.startswith('tfidf')]
metadata_features = top_features[~top_features['feature'].str.startswith('tfidf')]

print(f"\nTop metadata features: {metadata_features['feature'].head(5).tolist()}")
print(f"Number of important TF-IDF features: {len(tfidf_features)}")

# %%
# ============================================
# PART 12: INFERENCE FUNCTION
# ============================================

def predict_review(review_text, model, tfidf_vectorizer, scaler):
    """
    Predict if a single review is fake or genuine
    """
    # Create dataframe with single review
    single_review_df = pd.DataFrame({'review_text': [review_text]})
    
    # Extract features
    text_features = extract_text_features(review_text)
    
    # Preprocess text
    processed_text = preprocess_text(review_text)
    
    # TF-IDF transformation
    tfidf_features = tfidf_vectorizer.transform([processed_text])
    tfidf_df = pd.DataFrame(tfidf_features.toarray(), 
                            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
    
    # Prepare metadata features
    metadata_features = pd.DataFrame([text_features])
    metadata_cols = ['char_count', 'word_count', 'avg_word_length', 
                     'exclamation_count', 'capital_ratio', 'all_caps_words',
                     'sentiment_polarity', 'sentiment_subjectivity']
    metadata_features = metadata_features[metadata_cols]
    
    # Add dummy values for missing features
    if 'rating' in X_train.columns:
        metadata_features['rating'] = 3  # Default rating
    if 'verified_purchase' in X_train.columns:
        metadata_features['verified_purchase'] = 0
    if 'rating_deviation' in X_train.columns:
        metadata_features['rating_deviation'] = 0
    
    # Scale metadata
    metadata_scaled = scaler.transform(metadata_features)
    metadata_scaled_df = pd.DataFrame(metadata_scaled, columns=metadata_features.columns)
    
    # Combine features
    X_single = pd.concat([tfidf_df, metadata_scaled_df], axis=1)
    
    # Make prediction
    prediction = model.predict(X_single)[0]
    probability = model.predict_proba(X_single)[0]
    
    result = {
        'prediction': 'FAKE' if prediction == 1 else 'GENUINE',
        'fake_probability': probability[1],
        'genuine_probability': probability[0],
        'confidence': max(probability)
    }
    
    return result

# Test the inference function
test_reviews = [
    "This product is absolutely AMAZING!!! BEST PURCHASE EVER!!! BUY NOW!!!",
    "The product works as described. Good value for the price, though packaging could be better.",
    "DONT WASTE YOUR MONEY!!! TOTAL SCAM!!!",
    "I've been using this for a month now. It has pros and cons but overall satisfied with my purchase."
]

print("\n=== TESTING INFERENCE ===")
best_model = results[best_model_name]['model']

for review in test_reviews:
    result = predict_review(review, best_model, tfidf_vectorizer, scaler)
    print(f"\nReview: '{review[:50]}...'" if len(review) > 50 else f"\nReview: '{review}'")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Fake Probability: {result['fake_probability']:.2%}")

# %%
# ============================================
# PART 13: SAVE MODEL & COMPONENTS
# ============================================

import joblib

def save_model_pipeline(model, tfidf_vectorizer, scaler, model_name):
    """
    Save the trained model and preprocessing components
    """
    # Save model
    joblib.dump(model, f'{model_name}_model.pkl')
    
    # Save vectorizer
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
    
    # Save scaler
    joblib.dump(scaler, 'feature_scaler.pkl')
    
    print(f"\nModel pipeline saved successfully!")
    print(f"Files created:")
    print(f"  - {model_name}_model.pkl")
    print(f"  - tfidf_vectorizer.pkl")
    print(f"  - feature_scaler.pkl")

# Save the best model
save_model_pipeline(best_model, tfidf_vectorizer, scaler, best_model_name.replace(' ', '_'))

# %%



