# pandas
import pandas as pd

# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# matplot and seaborn
import matplotlib.pyplot as plt
import seaborn as sns

from plot_feature_importance import PlotFeatureImportances

# Load the data from the Excel sheet
# data = pd.read_excel("./data/Questionnaire_main_indexes.xlsx")
data = pd.read_excel("./data/Questionnaire_Complete.xlsx")

target = 'MSProne_IDX'
features = data.drop(columns=[target, 'ID', 'MSpr_SUM_balanced_NaN', 'MSpr_SUM']).select_dtypes(include=['number']).columns

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

classifiers = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Support Vector Machine': SVC(random_state=42),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=3)
}

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('classifier', classifiers['KNN'])
])

# Train the model
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification report for more detailed metrics
print(classification_report(y_test, y_pred, zero_division=1))

rf_model = pipeline.named_steps['classifier']

# PlotFeatureImportances(features, rf_model.feature_importances_)
