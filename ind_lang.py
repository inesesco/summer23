import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
import seaborn as sns

# Load the Excel data into a DataFrame
df = pd.read_excel('LEP_Summary.xlsx')

df_filtered = df[df['Number of Distinct Members (#)'] >= 32]

# Visualize the differences using bar plots
plt.figure(figsize=(8, 6))

# Bar plot for Difference SUD
plt.subplot(211)
plt.bar(df_filtered['Primary Language'], df_filtered['SUD Difference in % w/ English Speakers'], color="purple")
plt.xlabel('Primary Language')
plt.ylim(-0.5, 0.5)
plt.title('Difference in % of patients with SUD by language')
plt.ylabel('Difference in % with English Speakers (SUD)')

# Bar plot for Difference Mental Health
plt.subplot(212)
plt.bar(df_filtered['Primary Language'], df_filtered['MH Difference in % w/ English Speakers'], color="pink")
plt.xlabel('Primary Language')
plt.ylabel('Difference in % with English Speakers (Mental Health)')
plt.ylim(-0.5, 0.5)
plt.title('Difference in % of patients with Mental Health Disorders by language')

plt.tight_layout()
plt.show()

# Filter out rows with the "normal" value (0)
sud_difference = df_filtered.loc[df_filtered['SUD Difference in % w/ English Speakers'] != 0, 'SUD Difference in % w/ English Speakers']
mental_health_difference = df_filtered.loc[df_filtered['MH Difference in % w/ English Speakers'] != 0, 'MH Difference in % w/ English Speakers']

print(f'sud_difference: {sud_difference}')
print(f'mental_health_difference{mental_health_difference}')

# Perform t-tests
t_statistic_sud, p_value_sud = ttest_1samp(sud_difference, 0)  # Compare against 0 as the null hypothesis
t_statistic_mental_health, p_value_mental_health = ttest_1samp(mental_health_difference, 0)  # Compare against 0 as the null hypothesis

# Print the t-test results
print("T-test results for Difference SUD:")
print("t-statistic:", t_statistic_sud)
print("p-value:", p_value_sud)

print("\nT-test results for Difference Mental Health:")
print("t-statistic:", t_statistic_mental_health)
print("p-value:", p_value_mental_health)



# create the sud and mental health columns but only when there is a significant number of people
#     tuple that has number of people that speak that language and the difference in percentage
#     choose to exclude the outliers that only have a few people
# use that to create the graphs and the stat analyses (pick random ish number but can change it)
