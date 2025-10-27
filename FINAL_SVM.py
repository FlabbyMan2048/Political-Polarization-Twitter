import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import ttest_ind
import os

# make outputs folder
os.makedirs("outputs", exist_ok=True)

print("SVM - Political Polarization Analysis")

# Load and preprocess data
print("Loading and preprocessing data")

df = pd.read_csv("Result.csv", low_memory=False)

# sample 15% of the data (svm is slow)
df = df.sample(frac=0.15, random_state=42)
print("Loaded", len(df), "records (15% sample)")

# drop unnecessary columns
df = df.drop(columns=["Unnamed: 0", "Time at origin (UNIX Epoch Timestamp)"])

# one-hot encode edge type
df = pd.get_dummies(df, columns=["Edge Type"])

# combine hashtags
hashtag_cols = ["List of hashtags"] + [f"Column{i}" for i in range(1, 16)]
df[hashtag_cols] = df[hashtag_cols].fillna("").astype(str)
df["combined_hashtags"] = df[hashtag_cols].agg(" ".join, axis=1)

# tf-idf
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["combined_hashtags"])
df["tfidf_score"] = tfidf_matrix.mean(axis=1).A1

# drop old hashtag columns
df = df.drop(columns=hashtag_cols + ["combined_hashtags"])

# one-hot encode node indices
df = pd.get_dummies(df, columns=["Target Node Index", "Source Node Index"])

# convert to int
for c in df.columns:
    if "Target Node Index_" in c or "Source Node Index_" in c:
        df[c] = df[c].astype(int)

# scale features
cols_to_scale = ["Number of hyperlinks contained in the text", "Edge Type_reply",
                 "Edge Type_retweet", "tfidf_score"] + \
                 [c for c in df.columns if "Target Node Index_" in c or "Source Node Index_" in c]

scaler = StandardScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# rename classes
df["Political Polarity"] = df["Political Polarity"].replace({"-": 0, "left": 1, "right": 2})

# features and labels
X = df.drop(columns=["Political Polarity"])
y = df["Political Polarity"]

print("Features:", X.shape[1])
print("Samples:", X.shape[0])
print("Class distribution:")
print(y.value_counts())

# 5-fold cross validation
print("Cross-validation...")

svm = SVC(C=10, kernel="linear", class_weight="balanced", random_state=42)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_accs = []
val_accs = []
train_preds = []
val_preds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
    X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

    svm.fit(X_train, y_train)
    y_train_pred = svm.predict(X_train)
    y_val_pred = svm.predict(X_val)

    train_preds.extend(y_train_pred)
    val_preds.extend(y_val_pred)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Fold {fold}: Train acc = {train_acc:.4f}, Val acc = {val_acc:.4f}")

print("Average train acc:", np.mean(train_accs))
print("Average val acc:", np.mean(val_accs))

# t-test
t_stat, p_val = ttest_ind(train_preds, val_preds, equal_var=False)
print("T-test between train and val predictions:")
print("t-stat:", t_stat)
print("p-val:", p_val)

if p_val < 0.05:
    print("Result: significant difference")
else:
    print("Result: not significant")

# Learning curve
print("Generating learning curve")

train_sizes, train_scores, val_scores = learning_curve(
    svm, X_train_full, y_train_full,
    cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring="accuracy", n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10,6))
plt.plot(train_sizes, train_mean, label="Train", color="blue")
plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.1)
plt.plot(train_sizes, val_mean, label="Validation", color="green")
plt.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.1)
plt.title("SVM Learning Curve")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/svm_learning_curve.png", dpi=300)
plt.show()

print("Learning curve saved.")

# Final test evaluation
print("Evaluating on test set")

svm.fit(X_train_full, y_train_full)
y_test_pred = svm.predict(X_test)

test_acc = accuracy_score(y_test, y_test_pred)
print("Test accuracy:", test_acc)

print("Classification report:")
print(classification_report(y_test, y_test_pred, target_names=["Neutral", "Left", "Right"]))

cm = confusion_matrix(y_test, y_test_pred)
print("Confusion matrix:")
print(cm)

ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Neutral", "Left", "Right"]).plot(cmap=plt.cm.Blues)
plt.title("SVM Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/svm_confusion_matrix.png", dpi=300)
plt.show()

