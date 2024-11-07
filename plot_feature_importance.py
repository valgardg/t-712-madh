# pandas
import pandas as pd

# matplot and seaborn
import matplotlib.pyplot as plt
import seaborn as sns

def PlotFeatureImportances(features, importances, n=None):
    # Plot feature importances
    features_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    features_df = features_df.sort_values(by='Importance', ascending=False)
    if n is not None:
        features_df = features_df.nlargest(n, 'Importance')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=features_df)
    plt.title('Feature Importance in Decision Tree Model')
    plt.show()