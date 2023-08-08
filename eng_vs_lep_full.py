import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the Excel data into a DataFrame
df = pd.read_excel('LEP_Summary_Full.xlsx')

# Calculate percentages for each condition
conditions = ['SUD', 'Mental Health Disorder', 'OUD', 'IDD', 'HIV', 'Obesity']
for condition in conditions:
    df[f'% with {condition}'] = (df[f'# of Distinct Members with {condition} (#)'] / df['Number of Distinct Members (#)']) * 100

# Filter out languages with less than 32 members
df_filtered = df[df['Number of Distinct Members (#)'] >= 32]
df_filtered = df_filtered[df_filtered['Primary Language'] != "Total"]

# Define a function to calculate the threshold based on English speakers' percentage
def calculate_threshold(english_percentage):
    return 0.5 * english_percentage

# Perform calculations and create box plots for each condition
for condition in conditions:
    # Filter the DataFrame for "Other Languages" and English speakers for each condition
    english_speakers_condition = df_filtered.loc[df_filtered['Primary Language'] == 'ENGLISH', f'% with {condition}']
    other_languages_condition = df_filtered.loc[df_filtered['Primary Language'] != 'ENGLISH', f'% with {condition}']

    # Calculate the observed difference in percentages
    observed_diff = english_speakers_condition.item() - np.mean(other_languages_condition)

    # Perform permutation test
    num_permutations = 10000  # Number of permutations
    perm_diffs = np.zeros(num_permutations)  # Array to store the permuted differences

    # Concatenate English and non-English percentages
    all_percentages = np.concatenate((np.array([english_speakers_condition.item()]), other_languages_condition))

    for i in range(num_permutations):
        # Permute the labels (group assignments)
        permuted_labels = np.random.permutation(all_percentages)

        # Calculate the difference in means between permuted groups
        perm_diff = permuted_labels[0] - np.mean(permuted_labels[1:])

        # Store the permuted difference
        perm_diffs[i] = perm_diff

    # Calculate p-value as the proportion of permuted differences greater than or equal to observed difference
    p_value = np.mean(perm_diffs >= observed_diff)

    # Print the results
    print()
    print(f"Mean observed difference in percentages ({condition}):", observed_diff)
    print(f"p-value ({condition}):", p_value)

    # Calculate the threshold based on English speakers' percentage
    threshold = calculate_threshold(english_speakers_condition.item())

    # Filter languages with significant difference from English
    significant_languages = df_filtered[abs(df_filtered[f'% with {condition}']) >= abs(threshold)]
    
    # Create subplots with box plots side by side for each condition
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=[other_languages_condition, english_speakers_condition],
                flierprops=dict(marker='o', markersize=5, markerfacecolor='red', linestyle='none'),
                color="#253532")
    plt.xlabel('Language Group')
    plt.ylabel(f'% with {condition}')
    plt.title(f'Distribution of {condition} %: English Speakers vs. LEP')
    plt.xticks([0, 1], ['Other Languages', 'English Speakers'])

    # Add text labels for languages with significant difference from English
    significant_languages_list = []
    for i, row in significant_languages.iterrows():
        significant_languages_list.append((row['Primary Language'], row[f'% with {condition}']))
    print(f"\nLanguages with the highest difference in {condition} % compared to English speakers:")
    for language, difference in significant_languages_list:
        if language != 'ENGLISH':
            print(f"{language}: {difference:.2f}")

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
