"""
Created on Feb 25 00:39:02 2023
@author: JC Barajas
"""

import json
import re
import nltk
import emoji
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pysentimiento.preprocessing import preprocess_tweet


with open('data/SENT-COVID.json') as file:
    data = json.load(file)
    
pd.options.mode.chained_assignment = None                                        
pd.set_option('display.max_colwidth',None)   
df = pd.DataFrame(data)

def clean_tweet(text):
  text = re.sub(r'[~^0-9]', '', text) #numeros
  text = re.sub("\\s+", ' ', text) ##Espacios blancos dobles
  text = re.sub('\n', ' ', text) ##Saltos de linea

  pattern = r'([.])([A-Z#@¿])'
  pattern2 = r'([-?])([a-zA-Z#@¿])'
  pattern3 = r'([a-zA-Z])([#@¿(])'
  pattern4 = r'([:!])([a-zA-Z#@¿])'
  text = re.sub(pattern, r'\1 \2', text)  # Separacion de punto seguido por una mayuscula o signo '#', '@', '¿'
  text = re.sub(pattern2, r'\1 \2', text) # Separacion de '-' o '?' seguido por una letra o '#' o '@' o '¿'
  text = re.sub(pattern3, r'\1 \2', text)
  text = re.sub(pattern4, r'\1 \2', text)
  return text 


def preprocess(text):  # Preprocesamiento de pysentimiento   
  return preprocess_tweet(text, char_replace=True, normalize_laughter=True, shorten=2, 
                          emoji_wrapper='', user_token='', url_token='')  


def normalize(text):
 pattern2 = r'([a-zA-Z])([.])' # De haber separado antes estos patrones, no habria reconocido las url's.
 pattern3 = r'([.])([a-zA-Z])'
 text = re.sub(pattern2, r'\1 \2', text)
 text = re.sub(pattern3, r'\1 \2', text)
 
 text = "".join(u for u in text if u not in ("?","¿", ".", ";", ":", "!","¡",'"',"%","“","”","$","&","'","\\", "(",")",
                                             "*","+",",","/","<",">","=","^","•","...", "ç","π","ⓘ", "-", "_","#","|"))
 a,b = 'áéíóúÁÉÍÓÚ','aeiouAEIOU'
 trans = str.maketrans(a,b)     
 text = text.translate(trans) # Reemplazo de palabras acentuadas       

 pattern  = r'([a-z])([A-Z-])'
 text = re.sub(pattern, r'\1 \2', text)

 #text = re.sub(r'@[A-Za-z0-9_]+', '', text)
 text = text.lower()
 return text  


def tokenize(text):    
  text= text.split(sep = ' ')  # Tokenización por palabras individuales
  text= [token for token in text if len(token) > 1]  # Eliminación de tokens con una longitud < 2
  return(text) 

def labels(label):         # Converte etiquetas 'pos', 'neu', 'neg' en -1,0,1
  if label == 'POSITIVO':
    label=1
  elif label == 'NEUTRO':
     label=0
  else:
     label=-1
  return(label) 

