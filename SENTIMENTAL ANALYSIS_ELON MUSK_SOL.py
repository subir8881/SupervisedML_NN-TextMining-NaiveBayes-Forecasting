#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("Elon_musk.csv", error_bad_lines=False)
df.head()


# In[2]:


import re
# Clean The Data
def cleantext(text):
    text = re.sub(r"@[A-Za-z0-9]+", "", text) # Remove Mentions
    text = re.sub(r"#", "", text) # Remove Hashtags Symbol
    text = re.sub(r"RT[\s]+", "", text) # Remove Retweets
    text = re.sub(r"https?:\/\/\S+", "", text) # Remove The Hyper Link
    
    return text
# Clean The Text
df["Text"] = df["Text"].apply(cleantext)
df.head()


# In[3]:


from textblob import TextBlob
# Get The Subjectivity
def sentiment_analysis(ds):
    sentiment = TextBlob(ds["Text"]).sentiment
    return pd.Series([sentiment.subjectivity, sentiment.polarity])
# Adding Subjectivity & Polarity
df[["subjectivity", "polarity"]] = df.apply(sentiment_analysis, axis=1)
df


# In[4]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud
allwords = " ".join([twts for twts in df["Text"]])
wordCloud = WordCloud(width = 1000, height = 1000, random_state = 21, max_font_size = 119).generate(allwords)
plt.figure(figsize=(20, 20), dpi=80)
plt.imshow(wordCloud, interpolation = "bilinear")
plt.axis("off")
plt.show()


# In[5]:


# Compute The Negative, Neutral, Positive Analysis
def analysis(score):
    if score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    else:
        return "Positive"
    
# Create a New Analysis Column
df["analysis"] = df["polarity"].apply(analysis)
# Print The Data
df


# In[6]:


positive_text = df[df['analysis'] == 'Positive']
negative_text = df[df['analysis'] == 'Negative']
print('positive tweets')
for i, row in positive_text[:5].iterrows():
  print(' -' + row['Text'])
print('negative tweets')
for i, row in negative_text[:5].iterrows():
  print(' -' + row['Text'])


# In[7]:


plt.figure(figsize=(10, 8))
for i in range(0, df.shape[0]):
    plt.scatter(df["polarity"][i], df["subjectivity"][i], color = "Red")
plt.title("Sentiment Analysis") # Add The Graph Title
plt.xlabel("Polarity") # Add The X-Label
plt.ylabel("Subjectivity") # Add The Y-Label
plt.show() # Showing The Graph


# In[9]:


len(positive_text) / len(negative_text)


# In[45]:





# In[46]:


# Joinining all the reviews into single paragraph 
Txt_rev_string = " ".join(df["Text"])


# In[47]:


# Removing unwanted symbols incase if exists
Txt_rev_string = re.sub("[^A-Za-z" "]+"," ",Txt_rev_string).lower()
Txt_rev_string = re.sub("[0-9" "]+"," ",Txt_rev_string)


# In[48]:


Txt_rev_string


# In[49]:


import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import nltk
nltk.download('stopwords')


# In[50]:


# words that contained in Elon Musk reviews
Txt_reviews_words = Txt_rev_string.split(" ")
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

with open("stop.txt","r") as sw:
    stopwords = sw.read()

stopwords = stopwords.split("\n")


# In[51]:


stop_words


# In[52]:


temp = ["this","is","SpaceX","Production","Tesla"]
[i for i in temp if i not in "is"]

Txt_reviews_words = [w for w in Txt_reviews_words if not w in stopwords]


# In[53]:


Txt_reviews_words


# In[54]:


# Joinining all the reviews into single paragraph 
Txt_rev_string2 = " ".join(Txt_reviews_words)


# In[55]:


Txt_rev_string2


# In[56]:


# positive words # Choose the path for +ve words stored in system
with open("positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
  
poswords = poswords[36:]



# negative words  Choose path for -ve words stored in system
with open("negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")

negwords = negwords[37:]


# In[58]:


# negative word cloud
# Choosing the only words which are present in negwords
Txt_neg_in_neg = " ".join ([w for w in Txt_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(Txt_neg_in_neg)

plt.imshow(wordcloud_neg_in_neg)


# In[59]:


# Positive word cloud
# Choosing the only words which are present in positive words
Txt_pos_in_pos = " ".join ([w for w in Txt_reviews_words if w in poswords])
wordcloud_pos_in_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(Txt_pos_in_pos)

plt.imshow(wordcloud_pos_in_pos)

nltk 


# In[61]:


# Unique words 
Text_unique_words = list(set(" ".join(df["Text"]).split(" ")))


# In[63]:


Text_unique_words


# In[ ]:





# In[ ]:





# In[ ]:




