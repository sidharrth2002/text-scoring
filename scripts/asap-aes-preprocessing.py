#!/usr/bin/env python
# coding: utf-8

# ### Preprocessing ASAP-AES
# 
# 1. Preprocessing
# 2. Calculating NLP features from Uto et al. (2020)

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


import spacy

nlp = spacy.load("en_core_web_sm")

set3 = pd.read_csv('set3_features.csv')
# set3['lemmatized'] = set3['essay'].apply(lambda x: ' '.join([token.lemma_ for token in nlp(x)]))


# In[2]:


data = pd.read_csv('training_set_rel3.tsv', sep='\t', encoding='ISO-8859-1')
data.head()


# In[3]:


set3 = data[data['essay_set'] == 3]
set4 = data[data['essay_set'] == 4]
set5 = data[data['essay_set'] == 5]
set6 = data[data['essay_set'] == 6]


# In[4]:


import re
import string
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import spacy
from spellchecker import SpellChecker
import textstat

nlp = spacy.load('en_core_web_sm')

spell = SpellChecker()
english = English()
tokenizer = Tokenizer(english.vocab)

def count_commas(text):
  count = 0  
  for i in range (0, len(text)):   
    if text[i] == ',':  
        count = count + 1
  return count

def count_exclamation_marks(text):
  count = 0 
  for i in range (0, len(text)):   
    if text[i] == '!':  
        count = count + 1
  return count

def count_question_marks(text):
    count = 0
    for i in range (0, len(text)):
        if text[i] == '?':  
            count = count + 1
    return count

def lemmatize(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc]

def average_word_length(text):
    sentences = [sent.text for sent in nlp(text).sents]
    length_words = 0
    total_words = 0
    for sentence in sentences:
        words = tokenizer(sentence)
        for word in words:
            length_words += len(word)
            total_words += 1
    return length_words / total_words

def average_sentence_length(text):
    sentences = [sent.text for sent in nlp(text).sents]
    length_sentences = 0
    total_sentences = 0
    for sentence in sentences:
        length_sentences += len(tokenizer(sentence))
        total_sentences += 1
    return length_sentences / total_sentences

def number_of_nouns(text):
    # find number of nouns in text
    doc = nlp(text)
    pos = [token.pos_ for token in doc]
    return pos.count('NOUN') + pos.count('PROPN')

def number_of_verbs(text):
    # find number of verbs in text
    doc = nlp(text)
    pos = [token.pos_ for token in doc]
    return pos.count('VERB')

def number_of_adverbs(text):
    # find number of adverbs in text
    doc = nlp(text)
    pos = [token.pos_ for token in doc]
    return pos.count('ADV')

def number_of_adjectives(text):
    # find number of adjectives in text
    doc = nlp(text)
    pos = [token.pos_ for token in doc]
    return pos.count('ADJ')

def number_of_conjunctions(text):
    # find number of conjunctions in text
    doc = nlp(text)
    pos = [token.pos_ for token in doc]
    return pos.count('CCONJ')

def number_of_spelling_errors(text):
    misspelled = spell.unknown([token.text for token in tokenizer(text)])
    return len(misspelled)

def num_stopwords(text):
    # find number of stopwords in text
    doc = nlp(text)
    stop_words = [token.text for token in doc if token.is_stop]
    return len(stop_words)


# ### Features
# 
# <img src="./features.jpeg" width="600">

# In[73]:


def generate_features(frame):
    data = frame.copy()
    # length-based features
    print('Calculating number of words...')
    data['num_words'] = data['essay'].apply(lambda x: len(x.split()))
    print('Calculating number of sentences...')
    data['num_sentences'] = data['essay'].apply(lambda x: len(list(nlp(x).sents)))
    print('Calculating number of lemmas...')
    data['num_lemmas'] = data['essay'].apply(lambda x: len(lemmatize(x)))
    print('Calculating number of commas...')
    data['num_commas'] = data['essay'].apply(lambda x: count_commas(x))
    print('Calculating number of exclamation marks...')
    data['num_exclamation_marks'] = data['essay'].apply(lambda x: count_exclamation_marks(x))
    print('Calculating number of question marks...')
    data['num_question_marks'] = data['essay'].apply(lambda x: count_question_marks(x))
    print('Calculating average word length...')
    data['average_word_length'] = data['essay'].apply(lambda x: average_word_length(x))
    print('Calculating average sentence length...')
    data['average_sentence_length'] = data['essay'].apply(lambda x: average_sentence_length(x))

    # synctatic features
    print('Calculating number of nouns...')
    data['num_nouns'] = data['essay'].apply(lambda x: number_of_nouns(x))
    print('Calculating number of verbs...')
    data['num_verbs'] = data['essay'].apply(lambda x: number_of_verbs(x))
    print('Calculating number of adjectives...')
    data['num_adjectives'] = data['essay'].apply(lambda x: number_of_adjectives(x))
    print('Calculating number of adverbs...')
    data['num_adverbs'] = data['essay'].apply(lambda x: number_of_adverbs(x))
    print('Calculating number of conjunctions...')
    data['num_conjunctions'] = data['essay'].apply(lambda x: number_of_conjunctions(x))

    # word-based features
    print('Calculating number of spelling errors...')
    data['num_spelling_errors'] = data['essay'].apply(lambda x: number_of_spelling_errors(x))
    print('Calculating number of stopwords...')
    data['num_stopwords'] = data['essay'].apply(lambda x: num_stopwords(x))

    # readability features
    print('Calculating readability features...')
    data['automated_readability_index'] = data['essay'].apply(lambda x: textstat.automated_readability_index(x))
    data['coleman_liau_index'] = data['essay'].apply(lambda x: textstat.coleman_liau_index(x))
    data['dale_chall_index'] = data['essay'].apply(lambda x: textstat.dale_chall_readability_score(x))
    data['difficult_word_count'] = data['essay'].apply(lambda x: textstat.difficult_words(x))
    data['flesch_kincaid_grade'] = data['essay'].apply(lambda x: textstat.flesch_kincaid_grade(x))
    data['gunning_fog'] = data['essay'].apply(lambda x: textstat.gunning_fog(x))
    data['linsear_write_formula'] = data['essay'].apply(lambda x: textstat.linsear_write_formula(x))
    data['smog_index'] = data['essay'].apply(lambda x: textstat.smog_index(x))
    data['syllables_count'] = data['essay'].apply(lambda x: textstat.syllable_count(x))
    
    print('done')
    return data


# ### Split into sets

# In[74]:


def split_in_sets(data):
    essay_sets = []
    min_scores = []
    max_scores = []
    for s in range(1,9):
        essay_set = data[data["essay_set"] == s]
        essay_set.dropna(axis=1, inplace=True)
        n, d = essay_set.shape
        set_scores = essay_set["domain1_score"]
        print ("Set", s, ": Essays = ", n , "\t Attributes = ", d)
        min_scores.append(set_scores.min())
        max_scores.append(set_scores.max())
        essay_sets.append(essay_set)
    return (essay_sets, min_scores, max_scores)


# In[75]:


essay_sets, data_min_scores, data_max_scores = split_in_sets(data)
set1, set2, set3, set4, set5, set6, set7, set8 = tuple(essay_sets)


# In[76]:


set3 = generate_features(set3)
set4 = generate_features(set4)
set5 = generate_features(set5)
set6 = generate_features(set6)


# In[77]:


set3.to_csv('set3_features.csv', index=False)
set4.to_csv('set4_features.csv', index=False)
set5.to_csv('set5_features.csv', index=False)
set6.to_csv('set6_features.csv', index=False)


# ### Keyword Selection

# In[4]:


def get_quotes(text):
    quotes = re.findall(r'"([^"]*)"', text)
    return quotes


# In[105]:


set3_keywords = set()
for i in set3['essay']:
    unicode_converted = i.replace("\x93", '"').replace("\x94", '"')
    quotes = get_quotes(unicode_converted)
    if len(quotes) > 0:
        print(quotes, "\n")
        set3_keywords.update(quotes)


# In[107]:


len(set3_keywords)


# In[1]:


set3quotes = ['traveling through the high deserts of California in June', 'brackish water faling somewhere in the neighborhood of two hundred degrees', 'sun was beginning to beat down', 'growing realization that I could drop from heatstroke on a gorgous day in June', 'fit the traditional definition of a ghost town', 'trying to keep my balance in my dehydrated state', 'flat road was replaced by short rolling hills', 'water bottle contained only a few tantalizing sips', 'tarlike substance followed by brackish water', 'no one in sight, not a building, car, or structure of any kind', 'wide rings of dried sweat circled my shirt']

set3essayquotes = ['enjoyed the serenity of an early-summer evening', 'thriving little spot at one time', 'hitting my water bottles pretty regularly', 'high deserts of California', 'somewhere in the neighborhood of two hundred degrees', 'flat road was replaced by short rolling hills', 'ROUGH ROAD AHEAD: DO NOT EXCEED POSTED SPEED LIMIT', 'water bottles contained only a few tantalizing sips', 'Wide rings of dried sweat circled my shirt', 'drop from heatstroke on a gorgeous day', 'no one in sight, not a building, car, or structure of any kind', 'long, crippling hill', 'checked my water supply', 'birds would pick me clean']

set3_keywords = set(set3quotes + set3essayquotes)


# In[5]:


list(set3_keywords)


# In[5]:


set4 = pd.read_csv('set4_features.csv')


# In[6]:


set4_keywords = set()

for i in set4['essay']:
    unicode_converted = i.replace("\x93", '"').replace("\x94", '"').replace("\x85",  '')
    quotes = get_quotes(unicode_converted)
    if len(quotes) > 0:
        print(quotes, "\n")
        set4_keywords.update(quotes)

set4_keywords


# In[30]:


set4essay_quotes = ['will take that test again', 'many of the things that she had thought of as strange', 'not like the kind we had before', 'I failed the test', 'rich sweet scent', 'when the snow melts', 'geese return', 'hibiscus is budding', 'gentle grandmother', 'distinctive V was etched against the evening sky', 'familar fragrance filled her lungs', 'could almost feel light strands of her grandmother long gray hair', 'attitude towards her new country and her driving test', 'hibiscus plant in the winter is not as beautiful in the bitter cold', 'adapts and survives', 'returns to its beautiful state in the spring', 'bitter about her new country and driving test', 'new start or new opportunity', 'memories of home', 'overcoming her obstacles', 'noticed tear stains on her a daughter cheeks and her puffy eyes', 'symbolize change and adoption', 'make it through the winter into the spring', 'life is blooming', 'she was still too shaky to say the words at home', 'bitter melon']


# In[29]:


set4[set4['domain1_score'] == 3]['essay'].sample(1).iloc[0]


# In[23]:


len(set4essay_quotes)


# In[31]:


set5 = pd.read_csv('set5_features.csv')


# In[32]:


set5_keywords = set()

for i in set5['essay']:
    unicode_converted = i.replace("\x93", '"').replace("\x94", '"').replace("\x85",  '')
    quotes = get_quotes(unicode_converted)
    if len(quotes) > 0:
        print(quotes, "\n")
        set5_keywords.update(quotes)

set5_keywords


# In[33]:


set5essay_quotes = ['always be grateful to my parents for their love and sacrifice', 'rich culinary skills', 'love of cooking', 'passionate Cuban music', 'aromas of the kitchen', 'innocence of childhood', 'congregation of family and friends', 'endless celebrations', 'our warm home', 'came together in great solidarity and friendship', 'close-knit community of honest, hardworking immigrants', 'kept their arms and door open to the many people we considered family', 'came selflessly', 'struggled both personally and financially', 'facing cultural hardships', 'overt racism was the norm', 'drove them to endure these hard times', 'their strength and perseverance', 'love and sacrifice', 'spirit of generosity impressed upon me at such an early age', 'demonstration of how important family and friends are', 'teachings have been basis of my life', 'warmth of the kitchen', 'humble house', 'not just scent and music but life and love', 'definition of family', 'never forget how my parents turned this simple house into a home']


# In[36]:


set6 = pd.read_csv('set6_features.csv')
set6['num_words'].mean()


# In[43]:


set6_keywords = set()

for i in set6[set6['domain1_score'] == 4]['essay']:
    unicode_converted = i.replace("\x93", '"').replace("\x94", '"').replace("\x85",  '')
    quotes = get_quotes(unicode_converted)
    if len(quotes) > 0:
        print(quotes, "\n")
        set6_keywords.update(quotes)

set6_keywords


# In[42]:


set6[set6['domain1_score'] == 4]['essay'].sample(1).iloc[0]


# In[44]:


set6essay_quotes = ['one of safety', 'dirigibles from outside of the United States used hydrogen instead of helium', 'nature itself', 'winds on top of the building were constantly shifting', 'violent air currents', 'law against airships flying too low over urban areas', 'moored in open landing fields', 'could be weighted down in the back with lead weights', 'dangling high above pedestrians on the street was neither practical nor safe', 'swivel around and around the mooring mast', 'how much worse that accident could have been', 'could not simply drop a mooring mast on top of the empire state building flat roof', 'stress of the dirigible load', 'mooring air ships to a fixed mast', 'neither practical nor safe']


# In[45]:


max([len(i.split()) for i in set6essay_quotes])


# In[3]:


import pandas as pd
set3 = pd.read_csv('set3_features.csv')
set4 = pd.read_csv('set4_features.csv')


# In[2]:


set3['domain1_score'].value_counts()


# In[4]:


set4['domain1_score'].value_counts()


# In[ ]:




