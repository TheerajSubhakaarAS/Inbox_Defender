#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


dataset = pd.read_csv('spam.csv',encoding='latin')


# In[3]:


dataset.info()


# In[4]:


dataset.rename(columns={'v1':'target','v2':'text'},inplace=True)


# In[5]:


dataset.head(5)


# ## EDA

# In[6]:


dataset.shape


# In[25]:


import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[10]:


from nltk import word_tokenize,sent_tokenize
dataset['no_of_character']=dataset['text'].apply(len)
dataset['no_of_sentence']=dataset.apply(lambda row:sent_tokenize(row['text']),axis=1).apply(len)
dataset['no_of_words']=dataset.apply(lambda row:word_tokenize(row['text']),axis=1).apply(len)

dataset.describe().T


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,4))
fg = sns.countplot(x= dataset["target"])
fg.set_title("Count Plot of Classes")
fg.set_xlabel("Classes")
fg.set_ylabel("Number of Data points")


# In[12]:


## check for collinearity between independent features
plt.figure(figsize=(8,4))
sns.pairplot(data=dataset,hue='target')
plt.show()


# In[13]:


plt.figure(figsize=(8, 4))


sns.histplot(dataset[dataset['target'] == "spam"]['no_of_character'], color='blue', label='spam', kde=True)

sns.histplot(dataset[dataset['target'] == "ham"]['no_of_character'], color='red', label='ham', kde=True)


plt.xlabel('Number of Characters', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Number of Characters by Target', fontsize=16, fontweight='bold')


plt.legend()


sns.set(style='whitegrid') 

plt.show()


# In[14]:


plt.figure(figsize=(8, 4))


sns.histplot(dataset[dataset['target'] == "spam"]['no_of_words'], color='blue', label='spam', kde=True)

sns.histplot(dataset[dataset['target'] == "ham"]['no_of_words'], color='red', label='ham', kde=True)


plt.xlabel('Number of Sentences', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Number of Sentence by Target', fontsize=16, fontweight='bold')


plt.legend()


sns.set(style='whitegrid') 

plt.show()


# In[13]:


# implementing removal of outlier using IQR method


# In[15]:


#25th Quantile and 75th Quantile
Q1 = dataset['no_of_character'].quantile(0.25)
Q2 = dataset['no_of_character'].quantile(0.75) 


# In[16]:


# Calculate IQR
IQR = Q2 - Q1

# Determine outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q2 + 1.5 * IQR


# In[17]:


dataset = dataset[(dataset['no_of_character'] >= lower_bound) & (dataset['no_of_character'] <= upper_bound)]


# In[18]:


dataset.shape


# In[19]:


## Data preprocessing

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from string import punctuation


snowball_stemmer = SnowballStemmer(language='english')

def transform_text(text):
    tokens = word_tokenize(text.lower())

    processed_tokens=[
        snowball_stemmer.stem(token) for token in tokens
        if token.isalnum() and token not in stopwords.words('english') and token not in punctuation
    ]

    return " ".join(processed_tokens)

    


# In[22]:


## sample input trial and error 

for i in dataset['text']:
    sam_text = i
    break
print(i)
transform_text(i)


# In[23]:


from nltk.stem.arlstem import ARLSTem
stemmer = ARLSTem()
stemmer.stem('Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...')


# In[26]:


from nltk.stem import WordNetLemmatizer as wnl
print(wnl().lemmatize('Go until jurong point, crazy.. Available going only in bugis n great world la e buffet... Cine there got amore wat...'))


# In[27]:


from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()
st.stem('Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...')


# In[28]:


dataset['processed_text'] = dataset['text'].apply(transform_text)
dataset.head()


# In[29]:


from wordcloud import WordCloud

wc = WordCloud(width = 250, height = 250, min_font_size = 10, background_color = 'white')
spam_wc = wc.generate(dataset[dataset['target'] =="ham"]['processed_text'].str.cat(sep = " "))
plt.figure(figsize = (15,6))
plt.imshow(spam_wc)
plt.show()


# In[30]:


from wordcloud import WordCloud

wc = WordCloud(width = 250, height = 250, min_font_size = 10, background_color = 'white')
spam_wc = wc.generate(dataset[dataset['target'] =="spam"]['processed_text'].str.cat(sep = " "))
plt.figure(figsize = (15,6))
plt.imshow(spam_wc)
plt.show()


# In[31]:


#detailed view on Spam and Ham words

#find top 10 words of spam

#extracting spam corpus

spam_corpus=[]
for sentence in dataset[dataset['target']=="spam"]['processed_text'].tolist():
    for word in sentence.split():
        spam_corpus.append(word)


# In[32]:


from collections import Counter

spam_filtered = pd.DataFrame(Counter(spam_corpus).most_common(10))
spam_filtered.head()


# In[33]:


sns.barplot(data=spam_filtered,x=spam_filtered[0],y=spam_filtered[1],palette='bright')
plt.xticks(rotation=90)
plt.show()


# In[34]:


#find top 10 words of not spam words

#extracting not spam corpus

not_spam_corpus=[]
for sentence in dataset[dataset['target']=="ham"]['processed_text'].tolist():
    for word in sentence.split():
        not_spam_corpus.append(word)

not_spam_filtered = pd.DataFrame(Counter(not_spam_corpus).most_common(10))
not_spam_filtered.head()

sns.barplot(data=not_spam_filtered,x=not_spam_filtered[0],y=not_spam_filtered[1],palette='bright')
plt.xticks(rotation=90)
plt.show()


# In[35]:


## vectorizing method

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

cv = CountVectorizer()
tfid = TfidfVectorizer(max_features  = 3000)


# In[36]:


X = tfid.fit_transform(dataset['processed_text']).toarray()
from sklearn.preprocessing import LabelEncoder

dataset['target'] = LabelEncoder().fit_transform(dataset['target']) 
y = dataset['target'].values


# In[37]:


from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 42)


# In[38]:


from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# In[39]:


svc = SVC(kernel= "rbf", gamma  = 1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
rfc = RandomForestClassifier(n_estimators = 50, random_state = 2 )


# In[40]:


clfs = {
    'SVC': svc,
    'KNN': knc,
    'NB': mnb,
    'RF': rfc,    
}


# In[41]:


from sklearn.metrics import accuracy_score, precision_score, f1_score
import joblib

def train_classifier(clfs, X_train, y_train, X_test, y_test,model_path):
    clfs.fit(X_train,y_train)
    y_pred = clfs.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)
    joblib.dump(clfs, model_path)
    return accuracy , precision, f1score


# In[42]:


accuracy_scores = []
precision_scores = []
f1score=[]
for name , clfs in clfs.items():
    model_path = f"./models/{name}_model.pkl"

    current_accuracy, current_precision, current_f1score = train_classifier(clfs, X_train, y_train, X_test, y_test,model_path)
    print()
    print("For: ", name)
    print("Accuracy: ", current_accuracy)
    print("Precision: ", current_precision)
    print("F1-Score: ", current_f1score)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)
    f1score.append(current_f1score)

