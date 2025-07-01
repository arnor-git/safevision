#from google.colab import drive
#drive.mount('/content/drive')

import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import datetime
import json
import  os
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import matplotlib.patches as patches
from datetime import datetime
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Data Preprocssing


# file_path = '/content/drive/MyDrive/ET Dataset/ETdata-romania-useful.xlsm'
# data = pd.read_excel(file_path, sheet_name=None)  # Load all sheets into a dictionary
#
# sheets_with_student_id = []
#
# # Iterate over each sheet
# for sheet_name, df in data.items():
#     # Add a new column 'Student_id' with the sheet name for all rows
#     df['Student_id'] = sheet_name
#
#     # Append the modified DataFrame to the list
#     sheets_with_student_id.append(df)
#
# # Concatenate all DataFrames into one
# merged_data = pd.concat(sheets_with_student_id, ignore_index=True)
#
# merged_data.head()

# import pandas as pd
# from scipy.stats import zscore
#
# def preprocessdataset(merged_ndddata):
#     # Remove unwanted rows
#     merged_ndddata = merged_ndddata.drop(merged_ndddata[merged_ndddata['SubjectName (SubjectID) - GameID'] == 'Timestamp'].index)
#     merged_ndddata = merged_ndddata.drop(merged_ndddata[merged_ndddata['SubjectName (SubjectID) - GameID'] == '1(1) - 1'].index)
#
#     # Rename columns
#     new_columns = ['Timestamp', 'X', 'Y', 'LeftPupilDia', 'RightPupilDia', 'HeadX', 'HeadY', 'HeadZ', 'Label', 'Student_id']
#     merged_ndddata.columns = new_columns
#     merged_ndddata = merged_ndddata.drop(merged_ndddata.index[0])  # Drop initial row if needed
#
#     # Remove rows containing specific strings in any column
#     merged_ndddata = merged_ndddata[~merged_ndddata.astype(str).apply(lambda x: x.str.contains(',', regex=False)).any(axis=1)]
#     merged_ndddata = merged_ndddata[~merged_ndddata['Timestamp'].str.contains('\)', na=False)]
#
#     # Reset index
#     merged_ndddata.reset_index(drop=True, inplace=True)
#
#     # Convert relevant columns to numeric, handling errors
#     numeric_columns = ['X', 'Y', 'LeftPupilDia', 'RightPupilDia', 'HeadX', 'HeadY', 'HeadZ']
#     merged_ndddata[numeric_columns] = merged_ndddata[numeric_columns].apply(pd.to_numeric, errors='coerce')
#
#     # Drop rows with NaN values introduced by conversions
#     merged_ndddata.dropna(subset=numeric_columns, inplace=True)
#
#     # Detect and remove outliers based on Z-score method
#     z_scores = merged_ndddata[numeric_columns].apply(zscore)
#     threshold = 3  # Define a threshold for outlier detection
#     merged_ndddata = merged_ndddata[(z_scores < threshold).all(axis=1)]
#
#     return merged_ndddata
#
# data=preprocessdataset(merged_data)
#
# new_columns = ['Student_id', 'Timestamp', 'X', 'Y', 'LeftPupilDia', 'RightPupilDia', 'HeadX', 'HeadY', 'HeadZ', 'Label']
#
# merged_data = data[new_columns]
# merged_data.head()

#merged_data.to_csv('/content/drive/MyDrive/ET Dataset/romania processed data with all sheets and studnet id.csv', index=False)

data = pd.read_csv('/content/drive/MyDrive/privacy of students/romania processed data with all sheets and studnet id.csv')
data.head()

df = pd.DataFrame(data)

def process_ids(id_str):
    parts = id_str.split('_', 2)
    student = f"student{parts[0]}"
    level = f"Level{parts[1]}"
    timestamp = parts[2]
    return f"{student}_{level}_{timestamp}"

df['Student_id'] = df['Student_id'].apply(process_ids)

df.Student_id.unique()

# Feature engineering: Extract student ID and game level
df[['student_id', 'game_level', 'timestamp']] = df['Student_id'].str.split('_', expand=True)
df['game_level'] = df['game_level'].str.extract('(\d+)').astype(int)
df['student_id'] = df['student_id'].str.extract('(\d+)').astype(int)

df.head()

df=df.drop(columns=['Student_id'])
df.head()
new_column_order = ['student_id', 'game_level', 'timestamp', 'Timestamp', 'X', 'Y',
                    'LeftPupilDia', 'RightPupilDia', 'HeadX', 'HeadY', 'HeadZ', 'Label']

df = df[new_column_order]
df.head()

from sklearn.preprocessing import MinMaxScaler

# Features to be normalized
features_to_normalize = ['X', 'Y', 'LeftPupilDia', 'RightPupilDia', 'HeadX', 'HeadY', 'HeadZ']

scaler = MinMaxScaler()

df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])
joblib.dump(scaler, '/content/drive/MyDrive/privacy of students/scaler.pkl')

# Display the rearranged DataFrame
df.head()

print(df.game_level.unique())
print(df.Label.unique())

df.game_level= df.game_level.replace({4: 3})

df.game_level.unique()


labelpredictiondf=df.drop(columns=['student_id','timestamp','Timestamp'])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

# first Train on levels 1 and 2, test on level 3
train_df = labelpredictiondf[labelpredictiondf['game_level'] == 1]
test_df = labelpredictiondf[labelpredictiondf['game_level'].isin([2, 3])]

train_df = train_df.drop(columns=['game_level'])
test_df = test_df.drop(columns=['game_level'])

print("Class distribution in training set:")
print(train_df['Label'].value_counts())
print("\nClass distribution in testing set:")
print(test_df['Label'].value_counts())

X_train = train_df.drop(columns=['Label'])
y_train = train_df['Label']
X_test = test_df.drop(columns=['Label'])
y_test = test_df['Label']

# --- Random Forest Classifier ---

clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
cv_rf = cross_val_score(clf_rf, X_train, y_train, cv=5)
print(f"RandomForest 5-Fold Cross-Validation Accuracy: {cv_rf.mean()}")
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("RandomForest Accuracy on Test Set:", accuracy_rf)
print("RandomForest Classification Report:\n", classification_report(y_test, y_pred_rf))

conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_rf, display_labels=clf_rf.classes_)
disp_rf.plot(cmap='Blues')
plt.title('RandomForest Confusion Matrix')
plt.show()

feature_importances_rf = pd.DataFrame({'Feature': X_train.columns, 'Importance': clf_rf.feature_importances_})
feature_importances_rf = feature_importances_rf.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances_rf)
plt.title('RandomForest Feature Importances')
plt.show()

dt_clf = DecisionTreeClassifier(random_state=42)

cv_dt = cross_val_score(dt_clf, X_train, y_train, cv=5)
print(f"DecisionTree 5-Fold Cross-Validation Accuracy: {cv_dt.mean()}")

# Fit model on entire training data
dt_clf.fit(X_train, y_train)

# Predictions using DecisionTreeClassifier
y_pred_dt = dt_clf.predict(X_test)

# Evaluation for DecisionTreeClassifier
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("DecisionTree Accuracy on Test Set:", accuracy_dt)
print("DecisionTree Classification Report:\n", classification_report(y_test, y_pred_dt))

# Confusion matrix for DecisionTreeClassifier
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_dt, display_labels=dt_clf.classes_)
disp_dt.plot(cmap='Reds')
plt.title('DecisionTree Confusion Matrix')
plt.show()

# Plot Decision Tree visualization
plt.figure(figsize=(26, 10))
plot_tree(dt_clf, feature_names=X_train.columns, class_names=dt_clf.classes_, filled=True, rounded=True, fontsize=7)
plt.title('Decision Tree Visualization')
plt.show()

joblib.dump(clf_rf, 'student_diagnosis_randomforest.pkl')
joblib.dump(dt_clf, 'student_diagnosis_decisiontree.pkl')


studentidpredictiondf=df.drop(columns=['Label','timestamp','Timestamp'])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

# Filter data: Train on levels 1 and 2, test on level 3
train_df = studentidpredictiondf[studentidpredictiondf['game_level'] == 1]
test_df = studentidpredictiondf[studentidpredictiondf['game_level'].isin([2, 3])]

# Drop game_level feature
train_df = train_df.drop(columns=['game_level'])
test_df = test_df.drop(columns=['game_level'])

# Show class distribution in training and testing sets
print("Class distribution in training set:")
print(train_df['student_id'].value_counts())
print("\nClass distribution in testing set:")
print(test_df['student_id'].value_counts())

# Feature and label separation
X_train = train_df.drop(columns=['student_id'])
y_train = train_df['student_id']
X_test = test_df.drop(columns=['student_id'])
y_test = test_df['student_id']

# --- Random Forest Classifier ---

clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 5-Fold Cross-validation for RandomForestClassifier
cv_rf = cross_val_score(clf_rf, X_train, y_train, cv=5)
print(f"RandomForest 5-Fold Cross-Validation Accuracy: {cv_rf.mean()}")
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("RandomForest Accuracy on Test Set:", accuracy_rf)
print("RandomForest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Confusion matrix for RandomForestClassifier
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_rf, display_labels=clf_rf.classes_)
disp_rf.plot(cmap='Blues')
plt.title('RandomForest Confusion Matrix')
plt.show()

# Feature importance plot for RandomForestClassifier
feature_importances_rf = pd.DataFrame({'Feature': X_train.columns, 'Importance': clf_rf.feature_importances_})
feature_importances_rf = feature_importances_rf.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances_rf)
plt.title('RandomForest Feature Importances')
plt.show()

# --- Decision Tree Classifier ---

# Initialize DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(random_state=42)

# 5-Fold Cross-validation for DecisionTreeClassifier
cv_dt = cross_val_score(dt_clf, X_train, y_train, cv=5)
print(f"DecisionTree 5-Fold Cross-Validation Accuracy: {cv_dt.mean()}")

# Fit model on entire training data
dt_clf.fit(X_train, y_train)

# Predictions using DecisionTreeClassifier
y_pred_dt = dt_clf.predict(X_test)

# Evaluation for DecisionTreeClassifier
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("DecisionTree Accuracy on Test Set:", accuracy_dt)
print("DecisionTree Classification Report:\n", classification_report(y_test, y_pred_dt))

# Confusion matrix for DecisionTreeClassifier
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_dt, display_labels=dt_clf.classes_)
disp_dt.plot(cmap='Reds')
plt.title('DecisionTree Confusion Matrix')
plt.show()

class_names = [str(class_name) for class_name in dt_clf.classes_]

# Plot Decision Tree visualization
plt.figure(figsize=(26, 10))
plot_tree(dt_clf, feature_names=X_train.columns, class_names=class_names, filled=True, rounded=True, fontsize=7)
plt.title('Decision Tree Visualization')
plt.show()

joblib.dump(clf_rf, 'student_diagnosis_randomforest.pkl')
joblib.dump(dt_clf, 'student_diagnosis_decisiontree.pkl')


randomstudentidpredictiondf=df.drop(columns=['Label','timestamp','Timestamp','game_level'])
randomstudentidpredictiondf.head()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

X = train_df.drop(columns=['student_id'])
y = train_df['student_id']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# --- Random Forest Classifier ---

# Initialize RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 5-Fold Cross-validation for RandomForestClassifier
cv_rf = cross_val_score(clf_rf, X_train, y_train, cv=5)
print(f"RandomForest 5-Fold Cross-Validation Accuracy: {cv_rf.mean()}")

# Fit model on entire training data
clf_rf.fit(X_train, y_train)

# Predictions using RandomForestClassifier
y_pred_rf = clf_rf.predict(X_test)

# Evaluation for RandomForestClassifier
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("RandomForest Accuracy on Test Set:", accuracy_rf)
print("RandomForest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Confusion matrix for RandomForestClassifier
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_rf, display_labels=clf_rf.classes_)
disp_rf.plot(cmap='Blues')
plt.title('RandomForest Confusion Matrix')
plt.show()

# Feature importance plot for RandomForestClassifier
feature_importances_rf = pd.DataFrame({'Feature': X_train.columns, 'Importance': clf_rf.feature_importances_})
feature_importances_rf = feature_importances_rf.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances_rf)
plt.title('RandomForest Feature Importances')
plt.show()

# --- Decision Tree Classifier ---

# Initialize DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(random_state=42)

# 5-Fold Cross-validation for DecisionTreeClassifier
cv_dt = cross_val_score(dt_clf, X_train, y_train, cv=5)
print(f"DecisionTree 5-Fold Cross-Validation Accuracy: {cv_dt.mean()}")

# Fit model on entire training data
dt_clf.fit(X_train, y_train)

# Predictions using DecisionTreeClassifier
y_pred_dt = dt_clf.predict(X_test)

# Evaluation for DecisionTreeClassifier
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("DecisionTree Accuracy on Test Set:", accuracy_dt)
print("DecisionTree Classification Report:\n", classification_report(y_test, y_pred_dt))

# Confusion matrix for DecisionTreeClassifier
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_dt, display_labels=dt_clf.classes_)
disp_dt.plot(cmap='Reds')
plt.title('DecisionTree Confusion Matrix')
plt.show()

class_names = [str(class_name) for class_name in dt_clf.classes_]

# Plot Decision Tree visualization
plt.figure(figsize=(26, 10))
plot_tree(dt_clf, feature_names=X_train.columns, class_names=class_names, filled=True, rounded=True, fontsize=7)
plt.title('Decision Tree Visualization')
plt.show()

joblib.dump(clf_rf, 'student_diagnosis_randomforest.pkl')
joblib.dump(dt_clf, 'student_diagnosis_decisiontree.pkl')

#Out of sample Student (new studnet)

outofsmaplestudentidpredictiondf=df.drop(columns=['Label','timestamp','Timestamp','game_level'])

outofsmaplestudentidpredictiondf.head()

outofsmaplestudentidpredictiondf.student_id.unique()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier
import joblib

np.random.seed(42)
n_students = 9

# For demonstration, treating the first 8 students as existing and the 9th as new
existing_df = outofsmaplestudentidpredictiondf.iloc[:8].copy()
new_student_df = outofsmaplestudentidpredictiondf.iloc[8:9].copy()

# For verification only (not used in prediction)
true_student_id = 9
new_student_df.loc[new_student_df.index[0], 'student_id'] = true_student_id

print("Dataset Structure:")
print(df.head())
print("\nNew student data:")
print(new_student_df.drop(columns=['student_id']))

# Prepare features for training
X_existing = existing_df.drop(columns=['student_id'])
y_existing = existing_df['student_id']
X_new_student = new_student_df.drop(columns=['student_id'])

# Standardize the features
scaler = StandardScaler()
X_existing_scaled = scaler.fit_transform(X_existing)
X_new_student_scaled = scaler.transform(X_new_student)

print("\n" + "="*50)
print("APPROACH 1: Generate a New Student ID")
print("="*50)

max_existing_id = y_existing.max()
new_student_id = max_existing_id + 1

print(f"Assigned new unique ID {new_student_id} to the new student")

print("\n" + "="*50)
print("APPROACH 2: Student Similarity Matching")
print("="*50)

distances = cdist(X_new_student_scaled, X_existing_scaled, metric='euclidean')[0]

# Finding the index of the most similar student
most_similar_idx = np.argmin(distances)
most_similar_student_id = y_existing.iloc[most_similar_idx]

print(f"New student is most similar to student with ID: {most_similar_student_id}")
print(f"Similarity distance: {distances[most_similar_idx]}")

print("\n" + "="*50)
print("APPROACH 3: Outlier Detection")
print("="*50)

isolation_forest = IsolationForest(contamination=0.1, random_state=42)
isolation_forest.fit(X_existing_scaled)

# Predict if new student is an outlier (-1) or not (1)
is_outlier = isolation_forest.predict(X_new_student_scaled)

if is_outlier[0] == -1:
    print("This appears to be a new student, assign a new ID")
    new_id = y_existing.max() + 1
    print(f"Suggested new ID: {new_id}")
else:
    # Find closest match among existing students
    distances = cdist(X_new_student_scaled, X_existing_scaled, metric='euclidean')[0]
    closest_match = y_existing.iloc[np.argmin(distances)]
    print(f"This student may be an existing student with ID: {closest_match}")

print("\n" + "="*50)
print("APPROACH 4: Clustering with ID Assignment")
print("="*50)

# Train a KNN classifier on existing student data
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_existing_scaled, y_existing)

predicted_id = knn.predict(X_new_student_scaled)[0]
nearest_distance = cdist(X_new_student_scaled, X_existing_scaled, metric='euclidean').min()
confidence_score = np.exp(-nearest_distance)  # Convert distance to similarity score

print(f"Most likely student ID: {predicted_id}")
print(f"Confidence score: {confidence_score:.4f} (higher is more confident)")

# If confidence is low, assign a new ID instead
confidence_threshold = 0.5
if confidence_score < confidence_threshold:
    new_id = y_existing.max() + 1
    print(f"Confidence too low (< {confidence_threshold}), assigning new ID: {new_id}")

print("\n" + "="*50)
print("APPROACH 5: Feature-Based ID Generation")
print("="*50)

feature_string = '_'.join([str(round(val, 2)) for val in X_new_student.values[0]])

# Generate a hash and convert to integer
feature_hash = hashlib.md5(feature_string.encode()).hexdigest()
hashed_id = int(feature_hash[:8], 16) % 10000

while hashed_id in y_existing.values:
    hashed_id += 1

print(f"Generated feature-based ID for new student: {hashed_id}")
print(f"Based on feature string: {feature_string}")

print("\n" + "="*50)
print("APPROACH 6: KMeans Clustering")
print("="*50)

# Determine optimal number of clusters using the elbow method
wcss = []
max_clusters = min(6, len(X_existing))
for i in range(1, max_clusters+1):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_existing_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, max_clusters+1), wcss, marker='o', linestyle='-')
plt.title('Elbow Method For Optimal Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # Within-Cluster Sum of Squares
plt.grid(True)
plt.show()

# Based on elbow method, choose optimal number of clusters
optimal_clusters = 2
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(X_existing_scaled)

# Predict the cluster for the new student
new_student_cluster = kmeans.predict(X_new_student_scaled)

# Get the distance of the new student to the closest cluster center
distances_to_centers = kmeans.transform(X_new_student_scaled)
distance_to_cluster_center = np.min(distances_to_centers)

# Print results
print(f"New student belongs to cluster: {new_student_cluster[0]}")
print(f"Distance to closest cluster center: {distance_to_cluster_center:.4f}")

# Defining a threshold for determining if a student is an outlier
threshold = np.percentile(kmeans.transform(X_existing_scaled).min(axis=1), 95)
print(f"Outlier threshold (95th percentile of existing distances): {threshold:.4f}")

if distance_to_cluster_center > threshold:
    print(f"The new student is far from the cluster centers and is likely an outlier (new/unseen student).")
    print(f"Recommended approach: Assign a new ID: {max_existing_id + 1}")
else:
    # Find students in the same cluster
    existing_clusters = kmeans.predict(X_existing_scaled)
    same_cluster_indices = np.where(existing_clusters == new_student_cluster[0])[0]
    same_cluster_ids = y_existing.iloc[same_cluster_indices].values

    print(f"The new student belongs to the same cluster as existing students with IDs: {same_cluster_ids}")
    print(f"Recommended approach: Consider one of these IDs or assign a new ID based on business rules")
from sklearn.decomposition import PCA
# Apply PCA to reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
X_existing_pca = pca.fit_transform(X_existing_scaled)
X_new_student_pca = pca.transform(X_new_student_scaled)

cluster_centers_pca = pca.transform(kmeans.cluster_centers_)

explained_variance = pca.explained_variance_ratio_
print(f"\nPCA explained variance ratio: {explained_variance}")

plt.figure(figsize=(12, 10))

existing_clusters = kmeans.predict(X_existing_scaled)
for cluster in np.unique(existing_clusters):
    cluster_indices = np.where(existing_clusters == cluster)[0]

    scatter = plt.scatter(
        X_existing_pca[cluster_indices, 0],
        X_existing_pca[cluster_indices, 1],
        label=f"Cluster {cluster}",
        alpha=0.7,
        s=100
    )

    for idx in cluster_indices:
        plt.annotate(
            f"ID: {int(y_existing.iloc[idx])}",
            (X_existing_pca[idx, 0], X_existing_pca[idx, 1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )

plt.scatter(
    X_new_student_pca[:, 0],
    X_new_student_pca[:, 1],
    label="New Student",
    color='red',
    marker='X',
    s=200
)
plt.annotate(
    "New Student",
    (X_new_student_pca[0, 0], X_new_student_pca[0, 1]),
    textcoords="offset points",
    xytext=(0, -15),
    ha='center',
    color='red',
    weight='bold'
)

# Plot cluster centers
plt.scatter(
    cluster_centers_pca[:, 0],
    cluster_centers_pca[:, 1],
    marker='o',
    color='green',
    label='Cluster Centers',
    s=200,
    edgecolors='black'
)

plt.title("Student Clustering Visualization (PCA)", fontsize=14)
plt.xlabel(f"Principal Component 1 ({explained_variance[0]:.2%} variance)", fontsize=12)
plt.ylabel(f"Principal Component 2 ({explained_variance[1]:.2%} variance)", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


joblib.dump(kmeans, 'student_kmeans_clustering_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(isolation_forest, 'student_outlier_detection_model.pkl')
joblib.dump(knn, 'student_knn_classifier_model.pkl')

def predict_student_id(features_df, models_path='./'):
    scaler = joblib.load(f'{models_path}scaler.pkl')
    isolation_forest = joblib.load(f'{models_path}student_outlier_detection_model.pkl')
    knn = joblib.load(f'{models_path}student_knn_classifier_model.pkl')
    scaled_features = scaler.transform(features_df)
    is_outlier = isolation_forest.predict(scaled_features)[0] == -1
    nearest_dist = np.min(knn.kneighbors(scaled_features)[0])
    predicted_id = knn.predict(scaled_features)[0]
    confidence = np.exp(-nearest_dist)

    result = {
        'is_new_student': is_outlier,
        'predicted_id': predicted_id if not is_outlier else None,
        'confidence': confidence,
        'suggested_new_id': max(knn.classes_) + 1 if is_outlier else None,
        'nearest_neighbor_distance': nearest_dist
    }

    return result

