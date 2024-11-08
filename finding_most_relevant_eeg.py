from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd

df = pd.read_excel("./data/EEG.xlsx")

# Assuming you have your feature matrix X and target y
X = df.drop(columns='MSProne_IDX')  # Feature matrix (X)
y = df['MSProne_IDX']  # Target variable (y)

# Select top 10 features based on Chi-squared test (you can use other tests based on the data type)
selector = SelectKBest(score_func=chi2, k=10)
X_new = selector.fit_transform(X, y)

# Get the selected feature names
selected_features = pd.DataFrame(selector.get_support(indices=True), columns=["Feature Index"])
selected_feature_names = X.columns[selected_features["Feature Index"]]
print("Selected features:", selected_feature_names)
