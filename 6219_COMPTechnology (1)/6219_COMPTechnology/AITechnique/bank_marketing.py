# Bank Marketing Term Deposit Prediction - Random Forest
# Dataset: UCI Bank Marketing Dataset (bank-additional.csv, https://archive.ics.uci.edu/dataset/222/bank+marketing)
# Task: Predict whether a client will subscribe to a term deposit (y = yes/no)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("bank-additional.csv", sep=";")
print("Dataset shape:", df.shape)
print("\nTarget distribution:\n", df["y"].value_counts())

# Data preparation phase

# Replace "unknown" values with the column mode (most common value)
cat_cols = [c for c in df.select_dtypes(include="str").columns]
for col in cat_cols:
    df[col] = df[col].replace("unknown", df.loc[df[col] != "unknown", col].mode()[0])

# Encode education as numbers in order
edu_order = {
    "illiterate": 0,
    "basic.4y": 1,
    "basic.6y": 2,
    "basic.9y": 3,
    "high.school": 4,
    "professional.course": 5,
    "university.degree": 6
}
df["education"] = df["education"].map(edu_order).astype(int)

# Binary encode yes/no columns
for col in ["default", "housing", "loan", "y"]:
    df[col] = df[col].map({"no": 0, "yes": 1}).astype(int)

df["contact"] = df["contact"].map({"cellular": 1, "telephone": 0}).astype(int)

# Label encode remaining categorical columns
le = LabelEncoder()
for col in ["job", "marital", "month", "day_of_week", "poutcome"]:
    df[col] = le.fit_transform(df[col])

# Output our preprocessed dataset for any inspections
df.to_csv("bank-additional_preprocessed.csv", index=False)

# Features and target
X = df.drop("y", axis=1)
y = df["y"]

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid search to find best parameters
param_grid = {
    "n_estimators": [100, 300, 400, 500],
    "max_leaf_nodes": [8, 16, 32, 64, 128],
    "class_weight":     ["balanced", None],
    "min_samples_leaf": [1, 3, 5],
}

rf_clf = RandomForestClassifier(n_jobs=-1, random_state=42)
grid_search = GridSearchCV(rf_clf, param_grid, scoring="f1")
grid_search.fit(X_train, y_train)

print("\n==================\n")
print("Best Hyperparameters:", grid_search.best_params_)

# Get the best estimator
best_rf_clf = grid_search.best_estimator_
y_pred = best_rf_clf.predict(X_test)

print("\n==================\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature importances
print("Feature importances (higher = more important):")
for name, importance in zip(X.columns, best_rf_clf.feature_importances_):
    print(f"  {name}: {importance:.4f}")

# Plot feature importances and save to PNG
importances = pd.Series(best_rf_clf.feature_importances_, index=X.columns)
importances = importances.sort_values()

plt.figure(figsize=(8, 6))
plt.title("Feature Importances")
plt.xlabel("Importance")
importances.plot(kind="barh", color="steelblue")
plt.tight_layout()
plt.savefig("feature_importances.png")
print("\nPlot saved to feature_importances.png")

# Write predictions back to a CSV
# Re-read the original raw test rows (since we modified the last one) so the output is human-readable
original_df = pd.read_csv("bank-additional.csv", sep=";")
test_rows = original_df.iloc[X_test.index].copy()
test_rows["predicted_subscribe"] = y_pred
test_rows["actual_subscribe"] = y_test.values

# The contact list: rows the model says to call
contact_list = test_rows[test_rows["predicted_subscribe"] == 1].copy()
contact_list.to_csv("clients_to_contact.csv", index=False)

# Summary counts
total_test = len(test_rows)
total_actual = int(test_rows["actual_subscribe"].sum())
recommended = len(contact_list)
correctly_flagged = int(contact_list["actual_subscribe"].sum()) # true positives
false_alarms = recommended - correctly_flagged # false positives
missed = total_actual - correctly_flagged # false negatives
correctly_ignored = total_test - recommended - missed # true negatives

print(f"\nOut of {total_test} test clients:")
print(f"  Actually would subscribe     : {total_actual}")
print(f"  Model recommends contacting  : {recommended}")
print(f"  --- Of those recommended:")
print(f"      Would actually subscribe : {correctly_flagged}  (true positives)")
print(f"      Would NOT subscribe      : {false_alarms}  (false positives / wasted calls)")
print(f"  --- Of those NOT recommended:")
print(f"      Would actually subscribe : {missed}  (false negatives / missed revenue)")
print(f"      Would NOT subscribe      : {correctly_ignored}  (true negatives / correctly ignored)")
print("\nContact list saved to clients_to_contact.csv")

# Plot prediction outcome breakdown
labels = ["Correctly flagged\n(true positives)",
           "False alarms\n(false positives)",
           "Missed subscribers\n(false negatives)",
           "Correctly ignored\n(true negatives)"]
values = [correctly_flagged, false_alarms, missed, correctly_ignored]
colours = ["#2ecc71", "#e67e22", "#e74c3c", "#95a5a6"]

plt.figure(figsize=(8, 5))
plt.title("Model Prediction Outcomes (Test Set)")
bars = plt.bar(labels, values, color=colours, edgecolor="white")
plt.ylabel("Number of clients")
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             str(int(bar.get_height())), ha="center", fontsize=11)
plt.tight_layout()
plt.savefig("prediction_outcomes.png")
print("Plot saved to prediction_outcomes.png")
