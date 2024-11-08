import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns

#models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Assuming you have a DataFrame 'df' and 'MSProne_IDX' is your target column
df_complete = pd.read_excel("./data/Questionnaire_Complete.xlsx")
df_eeg = pd.read_excel("./data/EEG.xlsx")
df_emg = pd.read_excel("./data/EMG.xlsx")
df_cop = pd.read_excel("./data/Cop.xlsx")
data_main = pd.read_excel("./data/Questionnaire_main_indexes.xlsx")

df_complete_selected = df_complete[['Age_group', 'Gender', 'Height', 'Weight', 'BMI']]
df_eeg_selected = df_eeg[['THETA_frontal_relative_POST', 'DELTA_tempol_relative_75', 'LOWGAMMA_frontal_relative_PRE', 'BETA_frontal_relative_75', 'LOWGAMMA_tempol_relative_BASELINE', 'THETA_occipital_relative_50', 'LOWGAMMA_parietal_relative_BASELINE', 'ALPHA_frontal_relative_BASELINE']]
df_emg_selected = df_emg[['SL_EWL_25', 'TAL_FZC_25', 'SL_CV_75', 'TAL_EWL_50', 'TAL_LTKEO_BASELINE', 'TAL_EWL_BASELINE']]
df_cop_selected = df_cop[['Postero_Angle_75', 'MDIST_ML_cm__50', 'MDIST_ML_cm__POST', 'Left_Angle_25', 'MLCI_nats__50', 'DirectionEntropy_nats__PRE']]

# Combine dataframes - if they share a common identifier like "id", use it here in the merge
df = pd.concat(
    [df_eeg_selected, df_complete_selected, df_emg_selected, df_cop_selected],
    axis=1
)

# Print the assembled dataframe to verify
print(df)

target = 'MSProne_IDX'
df[target] = data_main[target]

# Prepare your feature matrix (X) and target variable (y)
# questionnaire_complete_drops = [target, 'ID', 'Self_assessed_MotionSickness_IDX', 'MSpr_SUM_balanced_NaN', 'MSpr_SUM', 'SUM_MS_POST']
# eeg_drops = [target, 'ID']
# emg_drops = [target, 'Unnamed: 0']
# cop_drops = [target, 'ID']
X = df.drop(columns=[target]).select_dtypes(include=['number'])  # Features (remove target)
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
    # ('feature_selector', RFE(estimator=RandomForestClassifier(random_state=42, class_weight='balanced'), n_features_to_select=10))
])

# Apply the pipeline to the training data
pipeline.fit(X_train, y_train)

remaining_features_after_variance = X_train.columns[pipeline.named_steps['variance_threshold'].get_support()]

# rfe_support_mask = pipeline.named_steps['feature_selector'].support_

# Step 5: Get the names of selected features after RFE
# selected_features_rfe = remaining_features_after_variance[rfe_support_mask]
# print("Selected features after RFE:", selected_features_rfe)

# Step 3: Transform the training and test sets using the pipeline
X_train_selected = pipeline.transform(X_train)
X_test_selected = pipeline.transform(X_test)

# Step 4: Train Random Forest Classifier on the selected features
rf_model = RandomForestClassifier(random_state=42,  class_weight='balanced')
rf_model.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test_selected)

# Calculate accuracy and display classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

y_prob = rf_model.predict_proba(X_test_selected)[:, 1]

# Step 2: Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Step 3: Calculate the AUC score
roc_auc = roc_auc_score(y_test, y_prob)

# Step 4: Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random chance
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()