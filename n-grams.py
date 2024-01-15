import pandas as pd
import re
import scipy.stats as stats
from statistics import mean, stdev
from math import sqrt
from nltk.util import ngrams
from pathlib import Path

base_path = Path(__file__).parent
file_path = (base_path / "../data/n_grams.csv").resolve()

print(file_path)
# Open the original transcription of the video
with open("data//original_text.md") as f:
    original_transcription = f.read()

# Select whitelisted characters a-z A-Z 0-9 and ' '
whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

# Clean original transcription
original_transcription = ''.join(filter(whitelist.__contains__, original_transcription)).lower()

# Create unigrams, bigrams and trigrams of original transcription
unigrams = set(list(ngrams(original_transcription.split(), 1)))
bigrams = set(list(ngrams(original_transcription.split(), 2)))
trigrams = set(list(ngrams(original_transcription.split(), 3)))

original_unigram_count = len(unigrams)
original_bigram_count = len(bigrams)
original_trigram_count = len(trigrams)

# Read note transcriptions
transcriptions = pd.read_csv("data//transcriptions.csv", encoding='unicode_escape')

# Clean note transcriptions
transcriptions['Notes'] = transcriptions.Notes.str.replace("[^a-zA-Z0-9 ]", "")

# Set columns for n-gram overlap percentages
transcriptions['unigram_percentage'] = 0.0
transcriptions['bigram_percentage'] = 0.0
transcriptions['trigram_percentage'] = 0.0
transcriptions['word_count'] = 0

# Iterate over dataframe to count uni-bi-and-trigrams
for index, row in transcriptions.iterrows():
    unigram_counter = 0
    bigram_counter = 0 
    trigram_counter = 0

    # Count total length of notes
    words = len(row['Notes'].split())
    transcriptions.at[index, 'word_count'] = words

    # Count unigrams
    for unigram in unigrams:
        if re.search(unigram[0], row['Notes']):
            unigram_counter += 1
    transcriptions.at[index, 'unigram_percentage'] = (unigram_counter/original_unigram_count) * 100

    # Count bigrams
    for bigram in bigrams:
        if re.search((bigram[0] + ' '+ bigram[1]), row['Notes']):
            bigram_counter += 1
    transcriptions.at[index, 'bigram_percentage'] = (bigram_counter/original_bigram_count) * 100

    # Count trigrams
    for trigram in trigrams:
        if re.search((trigram[0] + ' '+ trigram[1] + ' ' + trigram[2]), row['Notes']):
            trigram_counter += 1
    transcriptions.at[index, 'trigram_percentage'] = (trigram_counter/original_trigram_count) * 100

    print('Done with note nr: ', index +1)

# Save dataframe to csv
transcriptions.to_csv('data//n_grams.csv')

# Divide dataset into laptop notetakers & hand notetakers
data_laptop = transcriptions.loc[transcriptions['Laptop?']==1]
data_hand = transcriptions.loc[transcriptions['Laptop?']==0]

# Two sample t-test for unigrams: Statistically significant
unigram_means = transcriptions.groupby('Laptop?', as_index = False)['unigram_percentage'].mean()
print('\n')
print('Unigram means per condition:')
print(unigram_means)
print('\n')
print('Two-Sample T-test Unigrams: ')
print(stats.ttest_ind(a=data_laptop['unigram_percentage'], b=data_hand['unigram_percentage']))
cohens_d = (mean(data_laptop['unigram_percentage']) - mean(data_hand['unigram_percentage'])) / (sqrt((stdev(data_laptop['unigram_percentage']) ** 2 + stdev(data_hand['unigram_percentage']) ** 2) / 2))
print('Cohens d ', cohens_d)

# Two sample t-test for bigrams: Statistically significant
bigram_means = transcriptions.groupby('Laptop?', as_index = False)['bigram_percentage'].mean()
print('\n')
print('Bigram means per condition:')
print(bigram_means)
print('\n')
print('Two-Sample T-test bigrams: ')
print(stats.ttest_ind(a=data_laptop['bigram_percentage'], b=data_hand['bigram_percentage']))
cohens_d = (mean(data_laptop['bigram_percentage']) - mean(data_hand['bigram_percentage'])) / (sqrt((stdev(data_laptop['bigram_percentage']) ** 2 + stdev(data_hand['bigram_percentage']) ** 2) / 2))
print('Cohens d ', cohens_d)


# Two sample t-test for bigrams: Statistically significant
trigram_means = transcriptions.groupby('Laptop?', as_index = False)['trigram_percentage'].mean()
print('\n')
print('Trigram means per condition:')
print(trigram_means)
print('\n')
print('Two-Sample T-test trigrams: ')
print(stats.ttest_ind(a=data_laptop['trigram_percentage'], b=data_hand['trigram_percentage']))
cohens_d = (mean(data_laptop['trigram_percentage']) - mean(data_hand['trigram_percentage'])) / (sqrt((stdev(data_laptop['trigram_percentage']) ** 2 + stdev(data_hand['trigram_percentage']) ** 2) / 2))
print('Cohens d ', cohens_d)


# Two sample t-test for word count: Statistically significant
wordcount_means = transcriptions.groupby('Laptop?', as_index = False)['word_count'].mean()
print('\n')
print('Word count means per condition:')
print(wordcount_means)
print('\n')
print('Two-Sample T-test word counts: ')
print(stats.ttest_ind(a=data_laptop['word_count'], b=data_hand['word_count']))
cohens_d = (mean(data_laptop['word_count']) - mean(data_hand['word_count'])) / (sqrt((stdev(data_laptop['word_count']) ** 2 + stdev(data_hand['word_count']) ** 2) / 2))
print('Cohens d ', cohens_d)
