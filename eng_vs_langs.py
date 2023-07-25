import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Load the Excel data into a DataFrame
df = pd.read_excel('LEP_Summary.xlsx')

# Filter the DataFrame for languages other than English
other_languages_df = df[df['Primary Language'] != 'English']

# Get the percentage of members with SUD for English speakers
english_speakers_sud = df.loc[df['Primary Language'] == 'English', '% of members with SUD']


# Iterate over each language and perform the statistical test
for language, language_row in other_languages_df.iterrows():
    language_sud = language_row['% of members with SUD']
    
    # Perform the statistical test (t-test)
    t_statistic, p_value = ttest_ind([english_speakers_sud], [language_sud])
    
    print('Language:', language)
    print('Percentage of members with SUD:', language_sud)
    print('p-value:', p_value)
    print('---')
