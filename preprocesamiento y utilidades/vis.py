# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 00:39:02 2021
@author: Ricardo Lezama
"""

from gensim.models import Word2Vec

import numpy as np 
 
from nltk.corpus import stopwords 

import re 

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

plt.style.use('ggplot')


def normalize_corpus(raw_corpus):
    """
    This function reads in presumably somewhat clean. 
    
    Argument: a file path that contains a well formed txt file.
    
    Returns: This returns a 'list of lists' format friendly to Gensim. 
    """
    raw_corpus = open(raw_corpus,'r', encoding='utf-8').read().splitlines()
    #This is the simple way to remove stop words
    important_words=[]
    for sentences in raw_corpus:
        sentence = [] 
        a_words = re.findall(r'[A-Z,a-z,\-,0-9]+', sentences) 
        for a in a_words:
            if a not in stopwords.words('spanish') and ',' not in a:
                sentence.append(a.lower())
        important_words.append(sentence)
    return important_words
# Review words in an N dimensional vector space. 



def scatter_vector(model, palabra, size, topn):
    
    """ This scatter plot for vectors allows for quick visualization of similar terms. 
    
    Argument: a model containing vector representations of the Spanish MX content. word
    is the content you're looking for in the corpus.
    
    Return: close words as a print out. This also yields a plt.show() to visualize vectos in Anaconda.
    """
   
    arr = np.empty((0,size), dtype='f')
    word_labels = [palabra]
    palabras_cercanas = model.wv.similar_by_word(palabra, topn=topn)
    arr = np.append(arr, np.array([model.wv[palabra]]), axis=0)
    for wrd_score in palabras_cercanas:
        wrd_vector = model.wv[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()
    return palabras_cercanas    

important_text = normalize_corpus('C:/<<ZYZ>>/NER_news-main/corpora/todomexico.txt')

#Build the model, by selecting the parameters.
our_model = Word2Vec(important_text, vector_size=100, window=5, min_count=2, workers=20)
#Save the model
our_model.save("Mex_Corona_.w2v")
#Inspect the model by looking for the most similar words for a test word.
#print(our_model.wv.most_similar('mujeres', topn=5))
scatter_vector(our_model, 'Pfizer', 100, 21) 