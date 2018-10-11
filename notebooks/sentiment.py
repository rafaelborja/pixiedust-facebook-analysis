import json
import sys

import watson_developer_cloud
from watson_developer_cloud import ToneAnalyzerV3, VisualRecognitionV3
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions, EmotionOptions, SentimentOptions

import operator
from functools import reduce
from io import StringIO
import numpy as np
from bs4 import BeautifulSoup as bs
from operator import itemgetter
from os.path import join, dirname
import pandas as pd
import numpy as np
import requests
import pixiedust

# Suppress some pandas warnings
pd.options.mode.chained_assignment = None  # default='warn'
# Suppress SSL warnings
requests.packages.urllib3.disable_warnings()

# @hidden_cell

# Watson Visual Recognition
# VISUAL_RECOGNITION_API_KEY = '<add_vr_api_key>'

# Watson Natural Launguage Understanding (NLU)
NATURAL_LANGUAGE_UNDERSTANDING_USERNAME = 'f1e5dce1-3e78-43ba-bbcc-4c5ca3eefbf5'
NATURAL_LANGUAGE_UNDERSTANDING_PASSWORD = 'mgwU4ZEaUPYY'

# Watson Tone Analyzer
TONE_ANALYZER_USERNAME = '<add_tone_analyzer_username>'
TONE_ANALYZER_PASSWORD = '<add_tone_analyzer_password>'

# Create the Watson clients

nlu = watson_developer_cloud.NaturalLanguageUnderstandingV1(version='2017-02-27',
                                                            username=NATURAL_LANGUAGE_UNDERSTANDING_USERNAME,
                                                            password=NATURAL_LANGUAGE_UNDERSTANDING_PASSWORD)
# tone_analyzer = ToneAnalyzerV3(version='2016-05-19',
#                                username=TONE_ANALYZER_USERNAME,
#                                password=TONE_ANALYZER_PASSWORD)

# visual_recognition = VisualRecognitionV3(version='2018-05-22', iam_apikey=VISUAL_RECOGNITION_API_KEY)

# **Insert to code > Insert pandas DataFrame**
# df_data_1 = pd.read_csv('C:\\Users\\Rafael\\Downloads\\us-consumer-finance-complaint-database\\consumer_complaints_only_with_narrative_100.csv', encoding='latin-1')#, low_memory=False)
df_data_1 = pd.read_csv('C:\\Users\\Rafael\\Downloads\\us-consumer-finance-complaint-database\\consumer_complaints_only_with_narrative_100.csv', sep=',', error_bad_lines=False, index_col=False, dtype="unicode")

# Make sure this uses the variable above. The number will vary in the inserted code.
try:
    df = df_data_1
    print(df.head())
except NameError as e:
    print('Error: Setup is incorrect or incomplete.\n')
    print('Follow the instructions to insert the pandas DataFrame above, and edit to')
    print('make the generated df_data_# variable match the variable used here.')
    raise

# Make sure this uses the variable above. The number will vary in the inserted code.
# try:
#     credentials = credentials_1
# except NameError as e:
#     print('Error: Setup is incorrect or incomplete.\n')
#     print('Follow the instructions to insert the file credentials above, and edit to')
#     print('make the generated credentials_# variable match the variable used here.')
#     raise


df.rename(columns={'Post Message': 'consumer_complaint_narrative'}, inplace=True)


# Drop the rows that have NaN for the text.
df.dropna(subset=['consumer_complaint_narrative'], inplace=True)




df_http = df["consumer_complaint_narrative"].str.partition("http")
df_www = df["consumer_complaint_narrative"].str.partition("www")

# Combine delimiters with actual links
df_http["Link"] = df_http[1].map(str) + df_http[2]
df_www["Link1"] = df_www[1].map(str) + df_www[2]

# Include only Link columns
df_http.drop(df_http.columns[0:3], axis=1, inplace = True)
df_www.drop(df_www.columns[0:3], axis=1, inplace = True)

# Merge http and www DataFrames
dfmerge = pd.concat([df_http, df_www], axis=1)

# The following steps will allow you to merge data columns from the left to the right
dfmerge = dfmerge.apply(lambda x: x.str.strip()).replace('', np.nan)

# Use fillna to fill any blanks with the Link1 column
dfmerge["Link"].fillna(dfmerge["Link1"], inplace = True)

# Delete Link1 (www column)
dfmerge.drop("Link1", axis=1, inplace = True)

# Combine Link data frame
df = pd.concat([dfmerge,df], axis = 1)

# Make sure text column is a string
df["Text"] = df["consumer_complaint_narrative"].astype("str")

# Strip links from Text column
df['consumer_complaint_narrative'] = df['consumer_complaint_narrative'].apply(lambda x: x.split('http')[0])













# Define the list of features to get enrichment values for entities, keywords, emotion and sentiment
features = Features(entities=EntitiesOptions(), keywords=KeywordsOptions(), emotion=EmotionOptions(), sentiment=SentimentOptions())

#features = Features(sentiment=SentimentOptions())

overallSentimentScore = []
overallSentimentType = []
highestEmotion = []
highestEmotionScore = []
kywords = []
entities = []

# Go through every response and enrich the text using NLU.
for text in df['consumer_complaint_narrative']:
    print(text)
    # We are assuming English to avoid errors when the language cannot be detected.
    enriched_json = nlu.analyze(text=text, features=features, language='en').get_result()
    print(json.dumps(enriched_json))

    # Get the SENTIMENT score and type
    if 'sentiment' in enriched_json:
        if('score' in enriched_json['sentiment']["document"]):
            overallSentimentScore.append(enriched_json["sentiment"]["document"]["score"])
        else:
            overallSentimentScore.append('0')

        if('label' in enriched_json['sentiment']["document"]):
            overallSentimentType.append(enriched_json["sentiment"]["document"]["label"])
        else:
            overallSentimentType.append('0')
    else:
        overallSentimentScore.append('0')
        overallSentimentType.append('0')

    # Read the EMOTIONS into a dict and get the key (emotion) with maximum value
    if 'emotion' in enriched_json:
        me = max(enriched_json["emotion"]["document"]["emotion"].items(), key=operator.itemgetter(1))[0]
        highestEmotion.append(me)
        highestEmotionScore.append(enriched_json["emotion"]["document"]["emotion"][me])
    else:
        highestEmotion.append("")
        highestEmotionScore.append("")

    # Iterate and get KEYWORDS with a confidence of over 70%
    if 'keywords' in enriched_json:
        tmpkw = []
        for kw in enriched_json['keywords']:
            if(float(kw["relevance"]) >= 0.7):
                tmpkw.append(kw["text"])
        # Convert multiple keywords in a list to a string and append the string
        kywords.append(', '.join(tmpkw))
    else:
        kywords.append("")
            
    # Iterate and get Entities with a confidence of over 30%
    if 'entities' in enriched_json:
        tmpent = []
        for ent in enriched_json['entities']: 
            if(float(ent["relevance"]) >= 0.3):
                tmpent.append(ent["type"])
 
        # Convert multiple entities in a list to a string and append the string
        entities.append(', '.join(tmpent))
    else:
        entities.append("")    
    
# Create columns from the list and append to the DataFrame
if highestEmotion:
    df['TextHighestEmotion'] = highestEmotion
if highestEmotionScore:
    df['TextHighestEmotionScore'] = highestEmotionScore

if overallSentimentType:
    df['TextOverallSentimentType'] = overallSentimentType
if overallSentimentScore:
    df['TextOverallSentimentScore'] = overallSentimentScore

df['TextKeywords'] = kywords
df['TextEntities'] = entities






df.head()
print(df)

# Choose first of Keywords,Concepts, Entities
df["MaxTextKeywords"] = df["TextKeywords"].apply(lambda x: x.split(',')[0])
df["MaxTextEntity"] = df["TextEntities"].apply(lambda x: x.split(',')[0])