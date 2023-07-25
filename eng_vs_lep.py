import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel data into a DataFrame
df = pd.read_excel('LEP_Summary.xlsx')
df_filtered = df[df['Number of Distinct Members (#)'] >= 32]

#   SUD

# Calculate the observed difference in percentages
english_observed_sud = df_filtered.loc[df_filtered['Primary Language'] == 'ENGLISH', '% of members with SUD'].item()
non_english_percentages_sud = df_filtered.loc[df_filtered['Primary Language'] != 'ENGLISH', '% of members with SUD']

observed_diff_sud = english_observed_sud - np.mean(non_english_percentages_sud)

# Perform permutation test
num_permutations = 10000  # Number of permutations
perm_diffs_sud = np.zeros(num_permutations)  # Array to store the permuted differences

# Concatenate English and non-English percentages
all_percentages_sud = np.concatenate((np.array([english_observed_sud]), non_english_percentages_sud))

for i in range(num_permutations):
    # Permute the labels (group assignments)
    permuted_labels = np.random.permutation(all_percentages_sud)
    
    # Calculate the difference in means between permuted groups
    perm_diff = permuted_labels[0] - np.mean(permuted_labels[1:])
    
    # Store the permuted difference
    perm_diffs_sud[i] = perm_diff

# Calculate p-value as the proportion of permuted differences greater than or equal to observed difference
p_value_sud = np.mean(perm_diffs_sud >= observed_diff_sud)

# Print the results
print("Observed difference in percentages (SUD):", observed_diff_sud)
print("p-value (SUD):", p_value_sud)

#     MENTAL HEALTH DISORDERS

# Calculate the observed difference in percentages
english_observed_mhd = df_filtered.loc[df_filtered['Primary Language'] == 'ENGLISH', '% of members with Mental Health Disorders'].item()
non_english_percentages_mhd = df_filtered.loc[df_filtered['Primary Language'] != 'ENGLISH', '% of members with Mental Health Disorders']

observed_diff_mhd = english_observed_mhd - np.mean(non_english_percentages_mhd)

# Perform permutation test
num_permutations = 10000  # Number of permutations
perm_diffs_mhd = np.zeros(num_permutations)  # Array to store the permuted differences

# Concatenate English and non-English percentages
all_percentages_mhd = np.concatenate((np.array([english_observed_mhd]), non_english_percentages_mhd))

for i in range(num_permutations):
    # Permute the labels (group assignments)
    permuted_labels = np.random.permutation(all_percentages_mhd)
    
    # Calculate the difference in means between permuted groups
    perm_diff = permuted_labels[0] - np.mean(permuted_labels[1:])
    
    # Store the permuted difference
    perm_diffs_mhd[i] = perm_diff

# Calculate p-value as the proportion of permuted differences greater than or equal to observed difference
p_value_mhd = np.mean(perm_diffs_mhd >= observed_diff_mhd)

# Print the results
print("Observed difference in percentages (Mental Health Disorders):", observed_diff_mhd)
print("p-value (Mental Health Disorders):", p_value_mhd)

# Convert the percentage data to raw numbers by dividing by 100
df_filtered['% of members with SUD'] /= 100
df_filtered['% of members with Mental Health Disorders'] /= 100

# Filter the DataFrame to include only the "Non-English" languages
non_english = df_filtered[df_filtered['Primary Language'] != 'ENGLISH']

# Define a function to identify outliers using the IQR method
def find_outliers(column):
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    iqr = q3 - q1
    outlier_threshold = 1.5
    outliers = column[column < q1 - outlier_threshold * iqr]
    return outliers

# Identify outliers for SUD
outliers_sud = find_outliers(non_english['% of members with SUD'])

# Identify outliers for Mental Health Disorders
outliers_mental_health = find_outliers(non_english['% of members with Mental Health Disorders'])

# Print the outlier languages for SUD
print("Outlier languages for SUD:")
print(outliers_sud)

# Print the outlier languages for Mental Health Disorders
print("Outlier languages for Mental Health Disorders:")
print(outliers_mental_health)

# # Filter the DataFrame for "Other Languages" and English speakers
# other_languages_sud = df_filtered.loc[df_filtered['Primary Language'] != 'ENGLISH', '% of members with SUD']
# english_speakers_sud = df_filtered.loc[df_filtered['Primary Language'] == 'ENGLISH', '% of members with SUD']

# # Filter the DataFrame for "Other Languages" and English speakers
# other_languages_mhd = df_filtered.loc[df_filtered['Primary Language'] != 'ENGLISH', '% of members with Mental Health Disorders']
# english_speakers_mhd = df_filtered.loc[df_filtered['Primary Language'] == 'ENGLISH', '% of members with Mental Health Disorders']

# # Create subplots with two box plots side by side
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# # Box plot for SUD Percentages
# sns.boxplot(ax=axes[0], data=[other_languages_sud, english_speakers_sud], flierprops=dict(marker='o', markersize=5, markerfacecolor='red', linestyle='none'), color="purple")
# axes[0].set_xlabel('Language Group')
# axes[0].set_ylabel('% of members with SUD')
# axes[0].set_title('Distribution of SUD %: English Speakers vs. LEP')
# axes[0].set_xticklabels(['Other Languages', 'English Speakers'])

# # Box plot for Mental Health Disorders Percentages
# sns.boxplot(ax=axes[1], data=[other_languages_mhd, english_speakers_mhd], flierprops=dict(marker='o', markersize=5, markerfacecolor='red', linestyle='none'), color="pink")
# axes[1].set_xlabel('Language Group')
# axes[1].set_ylabel('% of members with Mental Health Disorders')
# axes[1].set_title('Distribution of Mental Health Disorders %: English Speakers vs. LEP')
# axes[1].set_xticklabels(['Other Languages', 'English Speakers'])

# # Adjust spacing between subplots
# plt.tight_layout()

# # Show the plot
# plt.show()