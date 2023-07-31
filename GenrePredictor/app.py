from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
np.random.seed(42)
from wordcloud import WordCloud
from collections import Counter
import re
import string


def preprocess(text):
    text = text.lower() #lowercase text
    text=text.strip()  #get rid of leading/trailing whitespace 
    text=re.compile('<.*?>').sub('', text) #Remove HTML tags/markups
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  #Replace punctuation with space. Careful since punctuation can sometime be useful
    text = re.sub('\s+', ' ', text)  #Remove extra space and tabs
    text = re.sub(r'\[[0-9]*\]',' ',text) #[0-9] matches any digit (0 to 10000...)
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) #matches any digit from 0 to 100000..., \D matches non-digits
    text = re.sub(r'\s+',' ',text) #\s matches any whitespace, \s+ matches multiple whitespace, \S matches non-whitespace 
    
    return text

#1. STOPWORD REMOVAL
from nltk.corpus import stopwords
def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)

#2. STEMMING
 
# Initialize the stemmer
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
snow = SnowballStemmer('english')
def stemming(string):
    a=[snow.stem(i) for i in word_tokenize(string) ]
    return " ".join(a)

#3. LEMMATIZATION
# Initialize the lemmatizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
wl = WordNetLemmatizer()
 
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Tokenize the sentence
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)

app = Flask(__name__)


def foo(x):
    return x.split()


model = pickle.load(open("SVC(class_weight='balanced').pkl",'rb'))


# @app.route('/')
# def index():
#     print("index start")
#     return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':
        sc = request.form['script']
        print(sc)
        print('-------------------LEN----------------------')
        print(len(sc))
        l1 = len(sc)
        print('-----------------------------------------')

        sc = preprocess(sc)
        print(sc)
        print('-------------------LEN----------------------')
        print(len(sc))
        print('-----------------------------------------')

        sc = stopword(sc)
        print(sc)
        print('-------------------LEN----------------------')
        print(len(sc))
        print('-----------------------------------------')

        sc = lemmatizer(sc)
        print(sc)
        print('-------------------LEN----------------------')
        print(len(sc))
        l2 = len(sc)
        print('-----------------------------------------')
        print("WORDS REDUCED FROM : ",l1,' TO : ',l2)
        print('-----------------------------------------')

        pred = model.predict([sc])
        return render_template('response.html',data=pred)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)