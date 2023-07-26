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

#     MENTAL HEALTH Conditions

# Calculate the observed difference in percentages
english_observed_mhc = df_filtered.loc[df_filtered['Primary Language'] == 'ENGLISH', '% of members with Mental Health Conditions'].item()
non_english_percentages_mhc = df_filtered.loc[df_filtered['Primary Language'] != 'ENGLISH', '% of members with Mental Health Conditions']

observed_diff_mhc = english_observed_mhc - np.mean(non_english_percentages_mhc)

# Perform permutation test
num_permutations = 10000  # Number of permutations
perm_diffs_mhc = np.zeros(num_permutations)  # Array to store the permuted differences

# Concatenate English and non-English percentages
all_percentages_mhc = np.concatenate((np.array([english_observed_mhc]), non_english_percentages_mhc))

for i in range(num_permutations):
    # Permute the labels (group assignments)
    permuted_labels = np.random.permutation(all_percentages_mhc)
    
    # Calculate the difference in means between permuted groups
    perm_diff = permuted_labels[0] - np.mean(permuted_labels[1:])
    
    # Store the permuted difference
    perm_diffs_mhc[i] = perm_diff

# Calculate p-value as the proportion of permuted differences greater than or equal to observed difference
p_value_mhc = np.mean(perm_diffs_mhc >= observed_diff_mhc)

# Print the results
print("Observed difference in percentages (Mental Health Conditions):", observed_diff_mhc)
print("p-value (Mental Health Conditions):", p_value_mhc)

# # Filter the DataFrame for "Other Languages" and English speakers
# other_languages_sud = df_filtered.loc[df_filtered['Primary Language'] != 'ENGLISH', '% of members with SUD']
# english_speakers_sud = df_filtered.loc[df_filtered['Primary Language'] == 'ENGLISH', '% of members with SUD']

# # Filter the DataFrame for "Other Languages" and English speakers
# other_languages_mhc = df_filtered.loc[df_filtered['Primary Language'] != 'ENGLISH', '% of members with Mental Health Conditions']
# english_speakers_mhc = df_filtered.loc[df_filtered['Primary Language'] == 'ENGLISH', '% of members with Mental Health Conditions']

# # Create subplots with two box plots side by side
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# # Box plot for SUD Percentages
# sns.boxplot(ax=axes[0], data=[other_languages_sud, english_speakers_sud], flierprops=dict(marker='o', markersize=5, markerfacecolor='red', linestyle='none'), color="purple")
# axes[0].set_xlabel('Language Group')
# axes[0].set_ylabel('% of members with SUD')
# axes[0].set_title(f'Dist. of % of Members with SUD (p-val: {p_value_sud})')
# axes[0].set_xticklabels(['Other Languages', 'English Speakers'])

# # Box plot for Mental Health Conditions Percentages
# sns.boxplot(ax=axes[1], data=[other_languages_mhc, english_speakers_mhc], flierprops=dict(marker='o', markersize=5, markerfacecolor='red', linestyle='none'), color="pink")
# axes[1].set_xlabel('Language Group')
# axes[1].set_ylabel('% of members with Mental Health Conditions')
# axes[1].set_title(f'Dist. of % of Members with Mental Health Conditions (p-val: {p_value_mhc})')
# axes[1].set_xticklabels(['Other Languages', 'English Speakers'])

# # Label the outliers on the SUD plot
# for ax in axes:
#     for i, artist in enumerate(ax.collections):
#         if i == 1:  # Check if it's the boxplot for English speakers
#             x = []
#             y = []
#             for path in artist.get_paths():
#                 y_values = path.vertices[:, 1]  # Get the y-values of the boxplot
#                 outliers = y_values[~np.logical_and(np.isfinite(y_values), np.isin(y_values, ax.get_ybound()))]
#                 x_values = np.full(len(outliers), i + 1)
#                 x.extend(x_values)
#                 y.extend(outliers)

#             ax.scatter(x, y, c='red', marker='o', s=50, label='Outliers')  # Set the label for the legend

# # Add a custom legend for outliers
# axes[1].legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Outliers')], loc='best')

# # Adjust spacing between subplots
# plt.tight_layout()

# # Show the plot
# plt.show()

# Filter the DataFrame for "Other Languages" and English speakers
other_languages_sud = df_filtered.loc[df_filtered['Primary Language'] != 'ENGLISH', '% of members with SUD']
english_speakers_sud = df_filtered.loc[df_filtered['Primary Language'] == 'ENGLISH', '% of members with SUD']

# Calculate the threshold for SUD based on half of the percentage of English speakers diagnosed with SUD
sud_threshold = abs(0.85 * english_speakers_sud.iloc[0])

# Filter languages with significant SUD difference from English
significant_sud_languages = df_filtered[abs(df_filtered['SUD Difference in % w/ English Speakers']) >= sud_threshold]

# Filter the DataFrame for "Other Languages" and English speakers
other_languages_mhc = df_filtered.loc[df_filtered['Primary Language'] != 'ENGLISH', '% of members with Mental Health Conditions']
english_speakers_mhc = df_filtered.loc[df_filtered['Primary Language'] == 'ENGLISH', '% of members with Mental Health Conditions']

# Calculate the threshold for Mental Health Conditions based on half of the percentage of English speakers diagnosed with Mental Health Conditions
mhc_threshold = abs(0.85 * english_speakers_mhc.iloc[0])

# Filter languages with significant Mental Health Conditions difference from English
significant_mhc_languages = df_filtered[abs(df_filtered['% of members with Mental Health Conditions']) >= mhc_threshold]

# Create subplots with two box plots side by side
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Box plot for SUD Percentages
sns.boxplot(ax=axes[0], data=[other_languages_sud, english_speakers_sud], flierprops=dict(marker='o', markersize=5, markerfacecolor='red', linestyle='none'), color="purple")
axes[0].set_xlabel('Language Group')
axes[0].set_ylabel('% of members with SUD')
axes[0].set_title('Distribution of SUD %: English Speakers vs. LEP')
axes[0].set_xticklabels(['Other Languages', 'English Speakers'])

# Box plot for Mental Health Conditions Percentages
sns.boxplot(ax=axes[1], data=[other_languages_mhc, english_speakers_mhc], flierprops=dict(marker='o', markersize=5, markerfacecolor='red', linestyle='none'), color="pink")
axes[1].set_xlabel('Language Group')
axes[1].set_ylabel('% of members with Mental Health Conditions')
axes[1].set_title('Distribution of Mental Health Conditions %: English Speakers vs. LEP')
axes[1].set_xticklabels(['Other Languages', 'English Speakers'])

# Add text labels for languages with significant SUD difference from English
significant_sud_languages_list = []
for i, row in significant_sud_languages.iterrows():
    significant_sud_languages_list.append((row['Primary Language'], row['% of members with SUD']))
print("Languages with the highest difference in SUD % compared to English speakers:")
for language, difference in significant_sud_languages_list:
    print(f"{language}: {difference:.2f}")

# Add text labels for languages with significant Mental Health Conditions difference from English
significant_mhc_languages_list = []
for i, row in significant_mhc_languages.iterrows():
    significant_mhc_languages_list.append((row['Primary Language'], row['% of members with Mental Health Conditions']))
print("\nLanguages with the highest difference in Mental Health Conditions % compared to English speakers:")
for language, difference in significant_mhc_languages_list:
    print(f"{language}: {difference:.2f}")

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()
