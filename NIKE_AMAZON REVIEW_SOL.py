#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
import nltk
from nltk.corpus import stopwords


# In[2]:


Nike_reviews=[]

### Extracting reviews from Amazon website ################
for i in range(1,10):
  ip =[]  
  url="https://www.amazon.in/Nike-Mens-Revolution-Running-Shoes/dp/B08Y5DYQXY/ref=sr_1_1?dchild=1&keywords=nike&qid=1628595068&sr=8-1"+str(i)
  response = requests.get(url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  reviews = soup.findAll("span",attrs={"class","a-size-base review-text"})# Extracting the content under specific tags  
  for i in range(len(reviews)):
    ip.append(reviews[i].text)  
  Nike_reviews=Nike_reviews+ip  # adding the reviews of one page to empty list which in future contains all the reviews

# writng reviews in a text file 
with open("Nike.txt","w",encoding='utf8') as output:
    output.write(str(Nike_reviews))
    


# In[3]:


cust_name = []   
review_title = []
rate = []
review_content = []


# In[4]:


tt = 0
while tt == 0:
    page = requests.get(url)
    while page.ok == False:
        page = requests.get(url)
   

    soup = bs(page.content,'html.parser')
    soup.prettify()      
    names = soup.find_all('span', class_='a-profile-name')
    names.pop(0)
    names.pop(0)
    
    for i in range(0,len(names)):
        cust_name.append(names[i].get_text())
        
    title = soup.find_all("a",{"data-hook":"review-title"})
    for i in range(0,len(title)):
        review_title.append(title[i].get_text())

    rating = soup.find_all('i',class_='review-rating')
    rating.pop(0)
    rating.pop(0)
    for i in range(0,len(rating)):
        rate.append(rating[i].get_text())

    review = soup.find_all("span",{"data-hook":"review-body"})
    for i in range(0,len(review)):
        review_content.append(review[i].get_text())
        
    try:
        for div in soup.findAll('li', attrs={'class':'a-last'}):
            A = div.find('a')['href']
        ul = bt + A
    except:
        break


# In[5]:


len(cust_name), len(review_title), len(review_content), len(rate)


# In[6]:


review_title[:] = [titles.lstrip('\n') for titles in review_title]

review_title[:] = [titles.rstrip('\n') for titles in review_title]

review_content[:] = [titles.lstrip('\n') for titles in review_content]

review_content[:] = [titles.rstrip('\n') for titles in review_content]


# In[7]:


df = pd.DataFrame()
df['Reviews'] = review_content


# In[8]:


df.head(5)


# In[9]:


df.dtypes


# In[10]:


df.shape


# In[11]:


df.describe()


# In[12]:


# Clean The Data
def cleantext(Reviews):
    Reviews = re.sub(r"@[A-Za-z0-9]+", "", Reviews) # Remove Mentions
    Reviews = re.sub(r"#", "", Reviews) # Remove Hashtags Symbol
    Reviews = re.sub(r"RT[\s]+", "", Reviews) # Remove Retweets
    Reviews = re.sub(r"https?:\/\/\S+", "", Reviews) # Remove The Hyper Link
    
    return Reviews
# Clean The Reviews
df["Reviews"] = df["Reviews"].apply(cleantext)
df.head()


# In[13]:


from textblob import TextBlob
# Get The Subjectivity
def sentiment_analysis(ds):
    sentiment = TextBlob(ds["Reviews"]).sentiment
    return pd.Series([sentiment.subjectivity, sentiment.polarity])
# Adding Subjectivity & Polarity
df[["subjectivity", "polarity"]] = df.apply(sentiment_analysis, axis=1)
df


# In[14]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud
allwords = " ".join([twts for twts in df["Reviews"]])
wordCloud = WordCloud(width = 1000, height = 1000, random_state = 21, max_font_size = 119).generate(allwords)
plt.figure(figsize=(20, 20), dpi=80)
plt.imshow(wordCloud, interpolation = "bilinear")
plt.axis("off")
plt.show()


# In[15]:


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


# In[16]:


positive = df[df['analysis'] == 'Positive']
negative = df[df['analysis'] == 'Negative']
print('positive tweets')
for i, row in positive[:5].iterrows():
  print(' -' + row['Reviews'])
print('negative Review')
for i, row in negative[:5].iterrows():
  print(' -' + row['Reviews'])


# In[17]:


plt.figure(figsize=(10, 8))
for i in range(0, df.shape[0]):
    plt.scatter(df["polarity"][i], df["subjectivity"][i], color = "Red")
plt.title("Sentiment Analysis") # Add The Graph Title
plt.xlabel("Polarity") # Add The X-Label
plt.ylabel("Subjectivity") # Add The Y-Label
plt.show() # Showing The Graph


# In[19]:


len(positive) / len(negative)


# In[20]:


nltk.download('wordnet')
from textblob import Word
df['Reviews']= df['Reviews'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

import re
pattern = r"((?<=^)|(?<= )).((?=$)|(?= ))"
df['Reviews']= df['Reviews'].apply(lambda x:(re.sub(pattern, '',x).strip()))


# In[21]:


df['Reviews'].head()

