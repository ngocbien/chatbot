from __future__ import unicode_literals, print_function, division
from nltk.corpus import stopwords
from nltk.stem import *
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("french", ignore_stopwords = False)
import csv
import re
from io import open
import unicodedata
import string
import re
import random
import os
import math
import time
from dateparser.search import search_dates
import pandas as pd

STOPWORDS = stopwords.words('french')
stemmer = SnowballStemmer("french", ignore_stopwords = False)

def date_time_():
    
    DICT_NUMERO = {'ce':1, 'cette':1, 'un':1, 'une':1, 'deux':2, 'trois':3,\
                   'quatre':4, 'cinq':5, 'six':6, 'sept':7,\
              'huit':8, 'neuf':9, 'dix':10, 'onze':11, 'douze':12, 'treize':13,'quartoze':14,\
               'quinze':15, 'seize':16, 'dix-sept':17,'dix sept':17, 'dix huit':18, \
               'dix-huit':18, 'dix-neuf':19, 'vingt':20, 'ving et un':21, 'trente':30, 'quarante':40,\
                  'quarante cinq':45, 'soixante':60}
    DICT_TEMPS = {'heure':1, 'matin':4, 'après-midi':2,  'journée':24,  'jour':24, \
              'mois':30*24, 'semaine':7*24, \
              'an':24*365, 'année':24*365, 'minute': 1/60}
    DICT_COMPLET_TEMPS = {}
    for temps in DICT_TEMPS:
        for numéro in DICT_NUMERO:
            key1 = numéro + ' '+temps
            key2 = str(DICT_NUMERO[numéro]) + ' ' +temps
            value = DICT_NUMERO[numéro]*DICT_TEMPS[temps]
            DICT_COMPLET_TEMPS[key1] = value
            DICT_COMPLET_TEMPS[key2] = value
    DICT_FOR_SPECIAL_CASE = {'aujourd\'hui':8, 'hébdomadaire':7*24, 'annuel': 365*24, \
                           'hier': 24, 'avant-hier': 48, 'annuelle': 365*24 , 'actuellement':1/2,\
                            'maintenant':1/2}
    DICT_COMPLET_TEMPS['#cas_particulier#'] = DICT_FOR_SPECIAL_CASE
    DICT_COMPLET_TEMPS['avant hier'] = 72
    DICT_COMPLET_TEMPS['semaine dernière'] = 24*7
    return DICT_COMPLET_TEMPS


def get_key_list(pairs_trains):
    
    dict_ = dict_manuelle()
    KEY_LIST = []
    KEY_LIST_ = []
    for pair in pairs_trains:
        for word in pair[1].split():
            if '#' in word:
                word = re.sub('#', '', word)
                if word not in KEY_LIST:
                    KEY_LIST.append(word)
    for key in KEY_LIST:
        key = stemmer.stem(key)
        if key not in KEY_LIST_ and '_' not in key:
            KEY_LIST_.append(key)
    KEY_RETURN = []
    for key in KEY_LIST_:
        boolean = False
        for list_ in dict_:
            if key in list_:
                KEY_RETURN.append(list_[0])
                boolean = True
                break
        if not boolean:
            KEY_RETURN.append(key)
    #for list_ in dict_:
       # if list_[0] not in KEY_RETURN:
        #    KEY_RETURN.append(list_[0])       
    return KEY_RETURN

def construct_list_city():
    #import csv
    liste_ville =[]
    with open('data/villes_france.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            liste_ville.append(row[2])
        for ville in liste_ville:
            if len(ville) <4:
                liste_ville.remove(ville)
    return liste_ville

def _find_right_time(question, Dict):
    
    list_ = ['?', '.', ' ', '\n', '!']
    for symbol in list_:
        question = question.lower().strip(symbol)
    list_word = question.split()
    word_ = None
    list_ = []
    if len(list_word) < 2:
        return None, None 
    for word in list_word:
        if word_ is not None:
            list_.append(str(word_)+' '+str(word))  
        word_ = word
    for key in Dict:
        for word in  list_:
            if key == word:
                return Dict[key], key
    for word in Dict['#cas_particulier#']:
        for word2 in question.split():
            if word2 == word:
                return Dict['#cas_particulier#'][word], word       
    return None, None


def _find_right_geography(question, Dict):
    """
    Nous cherchons une ville, un département dans le dictionnaire pour retourner soit
    ce nom de ce lieu, soit ses coordonnées correspondance. Le retour de ce donnée dépend 
    de quel façon nous allons traiter avec cette information.
    
    """
    list_ = ['?', '.', '\n', '!', ',']
    for symbol in list_:
        question = question.replace(symbol, '') 
    list_geography = []
    for word in question.split():
        if word.lower() !=word:
            word1 = word.lower()
            if word1 in Dict:
                list_geography.append(word)
    if len(list_geography) >0:
        return list_geography
    else:
        return None

def list_variable():
    """
    list_var is to manually complete """
    list_var = []
    if len(list_var) >0:
        return list_var
    else:
        return None

def _normalized_table():
    """
    return a data frame with new name
    """
    import pandas as pd
    df = pd.read_csv('fleet_donnees.csv', sep = ';')
    variable_names = list_variable()
    if variable_names ==None:
        return df
    else:
        name = list(df)
        dict_name = {name[i]: variable_names[i] for i in range(len(name))}
        df = df.rename(index= str, columns = dict_name)
        return df

def find_right_time_from_table(time, df):
    """
    given time found from a question, match it with the time given in table
    This depend on format of time in table.
    
    """
    return None
    
    
    
    
def clearing_word(word):
    
    word = re.sub('\x8e', 'é', word)
    word = re.sub('\x88', 'à', word)
    word = re.sub('\x9d', 'ù', word)
    word = re.sub('\x8f', 'è', word)
    word = re.sub('\x9e', 'û', word)
    word = re.sub('\x90', 'ê', word)
    word = re.sub('\x99', 'ô', word)
    word = re.sub('\x94', 'î', word)
   # word = re.sub('\x8f', 'è', word)
    word = re.sub('\x8d', 'ç', word)
    word = re.sub('õ', '', word)
    word = re.sub('Ê', '', word)
    word = re.sub('[?,.,!, \,,  %]', '', word)
    if word == 'û' or word == 'v' or word == 'é':
        word = ''
    if word =="2017êles":
        word = "2017"
    if "ênox" in word:
        word ="nox"
    return word


def clear_line(pharse):
    """
    Arg: string
    Return: string
    """
    dict_manuel = dict_manuelle()
    pharse_ =[]
    for words in pharse.strip().strip('?').strip('.').strip('!').lower().split(' '):
        for word in words.split('-'):
            for word_ in word.split('\''):
                word_ = clearing_word(word_)
                word_ = stemmer.stem(word_)
        #if word_ not in stop_word and len(word_) >1:       
        pharse_.append(word_)  
        for i in range(len(pharse_)):
            for line in dict_manuel:
                if pharse_[i] in line:
                    pharse_[i] = line[0]
    return ' '.join(pharse_)

def stemmer_line(line):
    
    return ' '.join([stemmer.stem(word) for word in line.split()])

def prepareData(PAIRS):
    
    pairs_trains, pairs_tests = [], []
   # PAIRS.extend(pairs1)
    index_train = random.sample(range(len(PAIRS)), int(len(PAIRS)*0.80))
    for i in range(len(PAIRS)):
        if i in index_train:
            pairs_trains.append(PAIRS[i])
        else:
            pairs_tests.append(PAIRS[i])
    print("Read %s sentence pairs of training set" % len(pairs_trains))
    print("Read %s sentence pairs of test set" % len(pairs_tests))
    return pairs_trains, pairs_tests


def get_all_convos(stopwords, file):
    
    convos = []
    questions_to_print = []
    liste_file = [file]
    list_of_word = []
    for file in liste_file:
        with open(file) as f:
            i=0
            for line in f:
                if i%2==0:
                    question = str(line)
                    if '++++' in question:
                         question = question[8:]
                    question_ = question.strip().strip('!').strip('?').strip('.')
                    questions_to_print.append(question_)
                    question = clear_line(question)
                else:
                    answer = str(line)
                    if '++++' in answer:
                        answer = answer[8:].strip()
                    convos.append([question, answer])
                i+=1
        f.close()
        good_index = []
        convos_ = []
        for i in range(len(convos)):
            if convos[i][0] not in convos_:
                convos_.append(convos[i][0])
                good_index.append(i)
            
    
    return [convos[i] for i in good_index], [questions_to_print[i] for i in good_index]

def convos_to_questions(convos):
    """
    prend tous les conversations dans corpus et renvoie un array de documents avec
    des vocabulaires séparés"""
    
    data = []
    for convo in convos:
        data.append(convo[0])
    return data

def split_words(doc):
    """
    Return word and term, for example, if document of 5 words, 
    it will return 5+4+3 words and terms"""
    list1 = doc.split()
    list2 = []
    for word in list1:
        list2.append(word)
    for i in range(len(list1)-1):
        word = list1[i]+ ' '+list1[i+1]
        list2.append(word)
    if len(list1)>2:
        for i in range(len(list1)-2):
            word = list1[i]+' '+list1[i+1]+list1[i+2]
            list2.append(word)
    return list2

def list_words(data):
    list_ = []
    for data_ in data:
        for term in split_words(data_):
            if term not in list_:
                list_.append(term)
    return list_
        

def data_processed():
    """
    Traiter et retourner les conversations avec les questions bien nettoyées
    les questions sans nettoyer et le couple pour test
    """
    convos1, questions1 = get_all_convos(STOPWORDS,'data/trains_sujet.txt')
    convos2, questions2  = get_all_convos(STOPWORDS, 'data/trains_hors_sujet.txt')
    convos3, questions3 = get_all_convos(STOPWORDS, 'data/convos3septembre_train.txt')
    convos1.extend(convos2)
    convos1.extend(convos3)
    questions1.extend(questions2)
    questions1.extend(questions3)
    convos_test1, questions_test1 = get_all_convos(STOPWORDS, 'data/tests_16juillet.txt')
    convos_test2, questions_test2 = get_all_convos(STOPWORDS, 'data/convos3septembre_test.txt')
    convos_test1.extend(convos_test2)
    questions_test1.extend(questions_test2)
    index_suff = [i for i in range(len(convos1))]
    random.shuffle(index_suff)
    convos, questions = [], []
    for i in index_suff:
        convos.append(convos1[i])
        questions.append(questions1[i])
    index_suff_ = [i for i in range(len(convos_test1))]
    random.shuffle(index_suff_)
    convos_test, questions_test = [], []
    for i in index_suff_:
        convos_test.append(convos_test1[i])
        questions_test.append(questions_test1[i])    
    return convos, questions, convos_test, questions_test

def dict_manuelle():
    """
    return la liste des listes des synonymes"""
    list_ = []
    dict_ = {}
    dict_['voiture'] = ['voiture', 'véhicule']
    dict_['km'] = ['km', 'kilomètrage', 'kilométrage', 'distance', 'kilomètre']
    dict_['essence'] = ['essence', 'carburant', 'énergie']
    dict_['roule'] = ['roule', 'roulent', 'marche', 'marcher', 'circule', 'circulent', 'circulation']
    dict_['bien'] = ['bien', 'bonne', 'good', 'excellent', 'parfait', 'beau', 'ok']
    dict_['min']  = ['min', 'minimum', 'minimal']
    dict_['pneu'] = ['pneumatique', 'pneu', 'pneus']
    dict_['pouquoi'] = ['cause', 'pourquoi', 'raison', 'motif']
    dict_['stupid'] = ['stupid', 'con', 'idiot', 'bête', 'imbécile']
    dict_['connaître'] = ['connaître', 'connaitre', 'savoir', 'comprendre',\
                          'connaisance', 'connais', 'connaîs', 'sais', 'savez']
    dict_['problème']  = ['problème', 'difficulté', 'danger', 'ennui', 'dangers']
    dict_['erreur'] = ['faute', 'erreur']
    dict_['arrêt'] = ['arrêt', 'arrêter', 'stopper', 'immobilisé', 'immobiliser', 'immobile', 'paralisé']
    dict_['disponible'] = ['disponible', 'disponibilité', 'libre']
    dict_['abimer'] = ['abîmer', 'hors service', 'cassé', 'endommager', 'abîmé']
    dict_['boitier'] = ['boitier', 'boîtier', 'boîte']
    dict_['frein'] = ['frein', 'freinage', 'freiner']
    dict_['parcouru'] = ['parcouru', 'parcourus', 'parcourir']
    dict_['peux'] = ['peux', 'pouvoir', 'pourrais']
    dict_['veux'] = ['veux', 'vouloir', 'veut', 'voulais', 'voudrais', 'souhaiter', 'souhaite']
    dict_['fait'] = ['fait', 'fais', 'faire', 'faites']
    dict_['fort'] = ['fort', 'puissant', 'robuste', 'solide']
    dict_['grand'] = ['grand', 'gross', 'élevé', 'haut', 'haute']
    dict_['mal'] = ['mal', 'mauvais', 'souci']
    dict_['changer'] = ['modifier', 'modification', 'changer', 'changement']
    dict_['avoir'] = ['as', 'obtenir', 'avoir', 'posséder', 'disposer']
    dict_['peur']  = ['peur', 'craint']
    dict_['placer'] = ['placer','installer']
    dict_['nombre'] = ['nombre', 'combien', 'quantité']
    dict_['aider']  = ['aider', 'aide', 'soutien', 'soutenir']
    dict_['question'] = ['question', 'demande']
    dict_['information'] = ['information', 'infos', 'renseignement', 'indictation',\
                            'informer', 'renseigner','indiquer']
    dict_['usé'] = ['usé', 'usure', 'vieux', 'vieille', 'fatigué']
    dict_['réduire']  = ['réduire', 'réduction', 'abaisser', 'diminuer', 'diminution']
    dict_['présent']  = ['présent', 'maitenant', 'actuel', 'actuellement']
    dict_['suivre']   = ['suivre', 'observer', 'surveiller', 'poursuivre']
    dict_['aller']    = ['aller', 'vas', 'vais', 'va']
    dict_['échanger'] = ['échanger', 'estimer', 'échangement', 'estimation',\
                         'évaluer', 'calculer','évaluation'\
                        , 'mesurer']
    dict_['taux']    = ['taux', 'pourcentage']
    dict_['ampoule'] = ['ampoule', 'feux', 'feu']
    for key in dict_:
        l = []
        for word in dict_[key]:
            l.append(stemmer.stem(word))
        list_.append(l)
    return list_

def dict_for_all_doc(data): 
    """
    return hold data's dictionary"""
    dict_ = {}
    for doc in data:
        list_word_ = split_words(doc)    
        dict_[doc] = {word: tf_idf_modified(word, doc, K) for word in list_word_}
    return dict_
