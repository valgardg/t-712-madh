import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have a DataFrame 'df' and 'MSProne_IDX' is your target column
# df = pd.read_excel("./data/Questionnaire_Complete.xlsx")
# df = pd.read_excel("./data/EEG.xlsx")
# df = pd.read_excel("./data/EMG.xlsx")
df = pd.read_excel("./data/Cop.xlsx")
data_main = pd.read_excel("./data/Questionnaire_main_indexes.xlsx")

target = 'MSProne_IDX'
df[target] = data_main[target]

# Prepare your feature matrix (X) and target variable (y)
# questionnaire_complete_drops = [target, 'ID', 'Self_assessed_MotionSickness_IDX', 'MSpr_SUM_balanced_NaN', 'MSpr_SUM', 'SUM_MS_POST']
# eeg_drops = [target, 'ID']
# emg_drops = [target, 'Unnamed: 0']
cop_drops = [target, 'ID']
X = df.drop(columns=cop_drops).select_dtypes(include=['number'])  # Features (remove target)
y = df[target]  # Target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1: Variance Threshold (Remove features with low variance)
variance_threshold = VarianceThreshold(threshold=0.01)
X_train_variance = variance_threshold.fit_transform(X_train)
X_test_variance = variance_threshold.transform(X_test)

# Keep track of remaining feature names after variance thresholding
remaining_features_after_variance = X_train.columns[variance_threshold.get_support()]

print(f"Remaining features after variance threshold: {X_train_variance.shape[1]}")
removed_features = X.columns[~variance_threshold.get_support()]
print("Features removed due to low variance:", removed_features)

# intermediary step: building pipeline
# Define the pipeline to handle variance threshold, imputation, and feature selection with RFE
pipeline = Pipeline([
    ('variance_threshold', VarianceThreshold(threshold=0.01)),
    ('imputer', SimpleImputer(strategy="mean")),
    ('feature_selector', RFE(estimator=RandomForestClassifier(class_weight='balanced'), n_features_to_select=10))
])

# Apply the pipeline to the training data
pipeline.fit(X_train, y_train)

remaining_features_after_variance = X_train.columns[pipeline.named_steps['variance_threshold'].get_support()]

rfe_support_mask = pipeline.named_steps['feature_selector'].support_

# Step 5: Get the names of selected features after RFE
selected_features_rfe = remaining_features_after_variance[rfe_support_mask]
print("Selected features after RFE:", selected_features_rfe)

# Step 3: Transform the training and test sets using the pipeline
X_train_selected = pipeline.transform(X_train)
X_test_selected = pipeline.transform(X_test)

# Step 4: Train Random Forest Classifier on the selected features
rf_model = RandomForestClassifier(class_weight='balanced')
rf_model.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test_selected)

# Calculate accuracy and display classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Get feature importances from the model
feature_importances = rf_model.feature_importances_

# Plot feature importances
features_df = pd.DataFrame({'Feature': selected_features_rfe, 'Importance': feature_importances})
features_df = features_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 4))
sns.barplot(x='Importance', y='Feature', data=features_df)
plt.title('Feature Importance in Random Forest Model')
plt.show()