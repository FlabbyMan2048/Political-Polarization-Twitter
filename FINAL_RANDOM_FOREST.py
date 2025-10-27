import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import os

# make outputs folder
os.makedirs("outputs", exist_ok=True)

print("Random Forest - Political Polarization Analysis")

# Load and preprocess data
print("Loading and preprocessing data...")

df = pd.read_csv("Result.csv", low_memory=False)

# sample 50% of data
df = df.sample(frac=0.5, random_state=42)
print("Loaded", len(df), "records")

# drop unnecessary columns
df = df.drop(columns=["Unnamed: 0", "Time at origin (UNIX Epoch Timestamp)"])

# one-hot encode edge type
df = pd.get_dummies(df, columns=["Edge Type"])

# combine hashtags and apply TF-IDF
hashtag_cols = ["List of hashtags"] + [f"Column{i}" for i in range(1, 16)]
df[hashtag_cols] = df[hashtag_cols].fillna("").astype(str)
df["combined_hashtags"] = df[hashtag_cols].agg(" ".join, axis=1)

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

# rename class labels
df["Political Polarity"] = df["Political Polarity"].replace({"-": 0, "left": 1, "right": 2})

# define X and y
X = df.drop(columns=["Number of hyperlinks contained in the text", "Political Polarity"])
y = df["Political Polarity"]

print("Features:", X.shape[1])
print("Samples:", X.shape[0])
print("Class distribution:")
print(y.value_counts())

# Cross-validation
print("5-fold cross validation")

rf = RandomForestClassifier(
    n_estimators=100,
    max_features="sqrt",
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=1,
    bootstrap=False,
    class_weight="balanced",
    random_state=42
)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_accs = []
val_accs = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full, y_train_full), 1):
    X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

    rf.fit(X_train, y_train)
    y_train_pred = rf.predict(X_train)
    y_val_pred = rf.predict(X_val)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Fold {fold}: Train = {train_acc:.4f}, Val = {val_acc:.4f}")

print("Average train accuracy:", np.mean(train_accs))
print("Average val accuracy:", np.mean(val_accs))

# Learning curve
print("Making a learning curve")

train_sizes = [0.1, 0.25, 0.5, 0.75]
train_means = []
val_means = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for size in train_sizes:
    temp_train = []
    temp_val = []
    for train_idx, val_idx in skf.split(X, y):
        X_train_cv, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val = y.iloc[train_idx], y.iloc[val_idx]
        sample_size = max(int(size * len(X_train_cv)), len(set(y)))
        X_sample, _, y_sample, _ = train_test_split(
            X_train_cv, y_train_cv, train_size=sample_size,
            random_state=42, stratify=y_train_cv
        )
        rf.fit(X_sample, y_sample)
        y_train_pred = rf.predict(X_sample)
        y_val_pred = rf.predict(X_val)
        temp_train.append(accuracy_score(y_sample, y_train_pred))
        temp_val.append(accuracy_score(y_val, y_val_pred))
    train_means.append(np.mean(temp_train))
    val_means.append(np.mean(temp_val))
    print(f"Size {int(size*100)}%: Train = {train_means[-1]:.4f}, Val = {val_means[-1]:.4f}")

# plot learning curve
plt.figure(figsize=(10,6))
actual_sizes = [int(size * len(X)) for size in train_sizes]
plt.plot(actual_sizes, train_means, label="Train", marker="o")
plt.plot(actual_sizes, val_means, label="Validation", marker="o")
plt.title("Random Forest Learning Curve")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/rf_learning_curve.png", dpi=300)
plt.show()
print("Learning curve saved.")

# Final test evaluation
print("Evaluating on test set")

rf.fit(X_train_full, y_train_full)
y_test_pred = rf.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)

print("Test accuracy:", test_acc)
print("Classification report:")
print(classification_report(y_test, y_test_pred, target_names=["Neutral", "Left", "Right"]))

cm = confusion_matrix(y_test, y_test_pred)
print("Confusion matrix:")
print(cm)

ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Neutral", "Left", "Right"]).plot(cmap=plt.cm.Blues)
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/rf_confusion_matrix.png", dpi=300)
plt.show()
print("Confusion matrix saved.")

# Feature importance
print("Checking feature importance")

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Top 10 features:")
for i in range(10):
    print(f"{i+1}. {X.columns[indices[i]]}: {importances[indices[i]]:.4f}")

top_n = 20
top_indices = indices[:top_n]
top_features = [X.columns[i] for i in top_indices]
top_values = importances[top_indices]

plt.figure(figsize=(10,8))
plt.barh(range(top_n), top_values, color="skyblue")
plt.yticks(range(top_n), top_features, fontsize=9)
plt.xlabel("Importance")
plt.title("Top 20 Features")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("outputs/rf_feature_importance.png", dpi=300)
plt.show()
print("Feature importance plot saved")


