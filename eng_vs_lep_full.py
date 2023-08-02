import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind

# Load the Excel data into a DataFrame
df = pd.read_excel('LEP_Summary_Full.xlsx')

# Calculate percentages for each condition
conditions = ['SUD', 'Mental Health Disorder', 'OUD', 'IDD', 'HIV', 'Obesity']
for condition in conditions:
    df[f'% with {condition}'] = (df[f'# of Distinct Members with {condition} (#)'] / df['Number of Distinct Members (#)']) * 100

# Filter out languages with less than 32 members
df_filtered = df[df['Number of Distinct Members (#)'] >= 32]

# Filter the DataFrame for "Other Languages" and English speakers for each condition
english_speakers_conditions = df_filtered.loc[df_filtered['Primary Language'] == 'ENGLISH', [f'% with {condition}' for condition in conditions]]
other_languages_conditions = df_filtered.loc[df_filtered['Primary Language'] != 'ENGLISH', [f'% with {condition}' for condition in conditions]]

# Calculate the threshold for each condition based on half of the percentage of English speakers diagnosed with that condition
thresholds = {condition: 0.5 * english_speakers_conditions[f'% with {condition}'].iloc[0] for condition in conditions}

# Perform bootstrap analysis for each condition
for condition in conditions:
    bootstrapped_differences = []
    english_observed = english_speakers_conditions[f'% with {condition}'].values

    for _ in range(10000):
        bootstrap_sample_english = np.random.choice(english_observed, size=len(other_languages_conditions), replace=True)
        bootstrapped_difference = bootstrap_sample_english.mean() - other_languages_conditions[f'% with {condition}'].mean()
        bootstrapped_differences.append(bootstrapped_difference)

    p_value = np.mean(bootstrapped_differences >= thresholds[condition])
    print(f"P-value for difference in % with {condition} between English speakers and Other Languages: {p_value:.4f}")

    # Create subplots with box plots side by side for each condition
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=[other_languages_conditions[f'% with {condition}'], english_speakers_conditions[f'% with {condition}']],
                flierprops=dict(marker='o', markersize=5, markerfacecolor='red', linestyle='none'),
                color="purple")
    plt.xlabel('Language Group')
    plt.ylabel(f'% with {condition}')
    plt.title(f'Distribution of {condition} %: English Speakers vs. LEP')
    plt.xticks([0, 1], ['Other Languages', 'English Speakers'])
    plt.tight_layout()

# Show the plots
plt.show()
