"""
les symboles particulières comme ? ! et . sont supprimé dans la question.
Il faut peut être returner une réponse spécifique si il rencontre plus 1 mot qu'il existe pas dans le corpus
Nous suprimmons également:
  1. Mot qui contient une seule lettre(qui sont suivante des erreurs)
  2. Une fonction unique pour nettoyage tous les données dans train-test et chat
Nous allons traiter tous les mots clés suivant :
km, vitesse, carburant, huile, position, batterie, pression_pneu. Il reste trajet et vin
  
"""
#sys
#!{sys.executable} -m pip uninstall gtts

#from gtts import gTTS
#from playsound import playsound
#import speech_recognition
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
import data_processed
import config


CONVOS_TRAIN, QUESTION, CONVOS_TEST, QUESTION_TEST= data_processed.data_processed()
CONVOS_TRAIN.extend(CONVOS_TEST)
DATA = data_processed.convos_to_questions(CONVOS_TRAIN)
#start = time.time()
#DICTIONARY =  data_processed.dict_for_all_doc(DATA)
KEY_LIST_  = data_processed.get_key_list(CONVOS_TRAIN)
COMPLET_DICT_TIME = data_processed.date_time_()
CITY_LIST = data_processed.construct_list_city()
LIST_OF_WORDS = data_processed.list_words(DATA)
K  = 1/10
STOPWORDS = stopwords.words('french')



def get_questions_for_classification():
    
    file1 = 'data/trains_sujet.txt'
    file2 = 'data/trains_hors_sujet.txt'
    liste_file = [file1, file2]
    index = 0
    for file in liste_file:
        questions = []
        with open(file) as f:
            i = 0 
            for line in f:
                if i%2 == 0:
                    question = str(line).strip().strip('?').strip('.').strip('!')
                    if '++++' in question:
                         question = question[8:]
                    questions.append(question)
                i += 1
        if index == 0:
            questions_sujet = questions
            index += 1
        else:
            questions_hors = questions
        f.close()
    return questions_sujet, questions_hors


def get_all_questions_to_print(file):
    
    convos = []
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
                    question = question.strip()
                    convos.append(question)
                i+=1
        f.close()
        indexes = []
        index = 0
    for question in convos:
        question = clear_line(question)
        if len(question) !=0:
                indexes.append(index)
        index +=1
    return [convos[i] for i in indexes]


def clear_pairs_trains(pairs_trains):
    questions = []
    index = []
    for i in range(len(pairs_trains)):
        if  pairs_trains[i][0] not in questions:
            index.append(i)
            questions.append(pairs_trains[i][0])
    return [pairs_trains[i] for i in index]

def tf_idf(word, doc):
    """
    Nous avons utilisé une normalisation pour TF-IDF"""
    tf, df = 0, 0
    len_doc = len(data_processed.split_words(doc))
    value_word = 1/math.sqrt(len_doc)
    value_word = math.sqrt(value_word)
    if word in str(doc):
        tf = value_word
    for doc_ in DATA:
        if word in str(doc_):
            df+=1
    if df>0:
        return tf*math.log(len(DATA)/(df), 10)
    else:
        return 0
    
def tf_idf_modified(word, doc, k):
    """
    Nous avons utilisé une normalisation pour TF-IDF"""
    tf, df = 0, 0
    len_doc = len(data_processed.split_words(doc))
    value_word = 1/math.sqrt(len_doc)
    value_word = math.sqrt(value_word)
    len_word = len(word.split())
    #len_word *= len_word
    if word in str(doc):
        tf = value_word*k/len_word
    for doc_ in DATA:
        if word in str(doc_):
            df+=1
    if df>0:
        return tf*math.log(len(DATA)/(df), 10)
    else:
        return 0

def dict_for_all_doc(data): 
    """
    return hold data's dictionary"""
    dict_ = {}
    for doc in data:
        list_word_ = data_processed.split_words(doc)    
        dict_[doc] = {word: tf_idf_modified(word, doc, K) for word in list_word_}
    return dict_
    
def dict_doc_score_for_classif(doc, sujet): 
    """
    We return score of all one words and all two consequent words"""

    list_word_ = data_processed.split_words(doc)    
    return {word: tf_idf_for_classif(word, doc, sujet) for word in list_word_}

   

        
        
        

def find_right_index(new_doc,DICT, méthode = 'normal'):
    
    index=0
    biggest_score = score_by_new_doc(new_doc, QUESTION_TO_TRAINS[0],DICT, méthode)
    for i in range(len(QUESTION_TO_TRAINS)):
        boolean1 = score_by_new_doc(new_doc, QUESTION_TO_TRAINS[i], DICT, méthode) >biggest_score
        boolean2 = score_by_new_doc(new_doc, QUESTION_TO_TRAINS[i], DICT, méthode) == biggest_score
        boolean3 = len(QUESTION_TO_TRAINS[index]) >len(QUESTION_TO_TRAINS[i])
        if boolean1 or (boolean2 and boolean3):
            biggest_score = score_by_new_doc(new_doc, QUESTION_TO_TRAINS[i],DICT, méthode)
            index = i
    précision, rappel = precision_recall(new_doc, QUESTION_TO_TRAINS[index])
    #if biggest_score ==0 or précision <0.2 or rappel<0.2:
     #   for question in QUESTION_TO_TRAINS:
      #      if '_return_' in question:
       #         try:
        #            index = QUESTION_TO_TRAINS.index(question)
         #       except ValueError:
          #          index = -1
    return index

def reformule_question_if_need(new_question, good_question):
    
    if 'to_return' in good_question:
        return True
    score1 = score_by_new_doc(new_question, good_question, DICT)
    score2 = 0
    for word in Dict_of_methode[good_question]:
        score2 += Dict_of_methode[good_question][word]
    if score1/score2 > 0.6:
        return  True
    else:
        return  False


def test(DICT, méthode, train =False):
    
    if not train:
        index_to_print = random.sample(range(len(pairs_tests)), 20)
        #loss_total = 0
        for i in range(len(pairs_tests)):
            index = find_right_index(pairs_tests[i][0], DICT, méthode)
            #loss = _evaluate_by_right_word(pairs_trains[index][1], pairs_tests[i][1])
            #loss_total +=loss
            if i in index_to_print:
                print(pairs_tests[i])
                print(pairs_trains[index][1])
        print('-'*80)
       # print('ACCURACY=', 1-loss_total/len(pairs_tests))
    if  train:
        #loss_total = 0
        #index_to_print = random.sample(range(len(pairs_trains)), 20)  
        list_of_bad_prediction = []
        i0 = 0
        for i in range(len(pairs_trains)):
            index = find_right_index(pairs_trains[i][0], DICT, méthode)
            #loss = _evaluate_by_right_word(pairs_trains[index][1], pairs_trains[i][1])
            #loss_total +=loss
            if index != i :
                i0 +=1
                print(pairs_trains[i])
                print(pairs_trains[index][1])
        print('-'*80)
        #print('ACCURACY=', 1-loss_total/len(pairs_trains))
        print('Il y a {} erreurs d\'indexes parmis {} prédictions'.format(i0, len(pairs_trains)))

        
def test_without_print(DICT, méthode, train =False):
    
    if not train:
        loss_total = 0
        for i in range(len(pairs_tests)):
            index = find_right_index(pairs_tests[i][0], DICT, méthode)
            loss = _evaluate_by_right_word(pairs_trains[index][1], pairs_tests[i][1])
            loss_total +=loss
        print('-'*80)
        print('ACCURACY=', 1-loss_total/len(pairs_tests))
    if  train:
        loss_total = 0
        index_to_print = random.sample(range(len(pairs_trains)), 20)   
        for i in range(len(pairs_trains)):
            index = find_right_index(pairs_trains[i][0], DICT, méthode)
            loss = _evaluate_by_right_word(pairs_trains[index][1], pairs_trains[i][1])
            loss_total +=loss
        print('-'*80)
        print('ACCURACY=', 1-loss_total/len(pairs_trains))

def change_subjet_of_question(question):
    dict_ = {'ton':'mon', 'ta':'ma', 'votre':'mon', 'tes':'mes', 'mon':'ton', \
            'ma':'ta', 'mes':'tes', 'tu': 'je', 'moi':'toi', 'je':'tu', 'toi':'moi',\
            'vous':'je'}
    question_ = []
    for word in question.lower().split():
        if word in dict_:
            word = dict_[word]
        question_.append(word)
    return ' '.join(question_)
        
def chat(méthode='normal'):
    
    print('Bonjour, C\'est le bot d\'Avicen, pose tes questions, s\'il te plaît!')
    path = 'processed/chat_tfidf.txt'
    f =  open(path, 'a+') 
    while True:
            line = str(input('Vous: '))
            list_city = _find_right_geography(line, CITY_LIST)
            line = line.lower().strip().strip('?').strip('.')
            time, time_word = _find_right_time(line, COMPLET_DICT_TIME)
            if time_word is not None:
                line = line.replace(time_word, '')
            if list_city is not None:
                for city in list_city:
                    line = line.replace(city, '')
            LINE = clear_line(line)
            unknown_word =0
            for word in LINE.split():
                if word not in LIST_OF_WORDS:
                    unknown_word += 1
            #if unknown_word > 1:
             #   print('Désolé, je ne comprends pas ta question, peux tu la reformuler s\'il te plaît?')
              #  continue
            if len(line) !=0:
                index= find_right_index(LINE, DICT, méthode)
                good_question = QUESTION_TO_TRAINS[index]
                if not reformule_question_if_need(LINE, good_question):
                    get_answer = str(input('Tu veux demander: {}?  |'.format(questions_sujet[index])))
                    if 'non' in get_answer.lower():
                        print('Peux tu reformuler ta question s\'il te plaît?')
                        continue
                if time == None and list_city == None:
                     print('bot: ', pairs_trains[index][1])
                elif list_city ==None:
                    print('Bot: {}, {}'.format(time_word, pairs_trains[index][1]))
                elif time == None:
                    cities = ' '.join(list_city)
                    print('Bot: À {}, {}'.format(cities, pairs_trains[index][1]))
                else:
                    cities = ' '.join(list_city)
                    print('Bot: À {}, {}, {}'.format(cities, time_word, pairs_trains[index][1]))
                f.write('VOUS ++++ '+line+'\n')
                f.write('BOT ++++ '+pairs_trains[index][1]+'\n')
            
            else:
                f.close()
                break 

def precision_recall(lstcomp, lstref):
    
    card_intersec = 0.0 # force à utiliser la division non entière
    for t in set(lstcomp) :
        card_intersec += min(lstref.count(t), lstcomp.count(t))
    if len(lstcomp)==0:
        precision =1
    else:
        precision = card_intersec/len(lstcomp)
    if len(lstref)==0:
        rappel = 1
    else:
        rappel = card_intersec/len(lstref)
    return (precision, rappel)


def calcul_precision_recall_(train = False): 
    
    if not train:
        total_precision, total_recall = 0, 0
        len_ = len(convos_test)
        index_to_print = random.sample(range(len_), 30)
        for i in range(len(questions_test)):
            question = convos_test[i][0]
            index = find_right_index_(question)
            answer = convos_train[index][1]
            good_answer = convos_test[i][1]
            precision, recall = precision_recall\
            (answer, good_answer)
            total_precision  += precision
            total_recall     += recall
            #if i in index_to_print:
            #    print('raw questions: {}. precision {}'.format( questions_test[i], precision))
             #   print('answer: {}. good answer: {} '.format(answer, good_answer))
        #print('-'*80)
        total_precision = total_precision/len_
        total_recall = total_recall/len_      
        F = 2*total_precision*total_recall/(total_precision+total_recall)
        print('ON TEST, WE HAVE: PRECISION = {:.5f},  RECALL = {:.5f}, F_MESURE = {:.3f} '\
              .format(total_precision, total_recall,F))
        print()
    if  train:       
        total_precision, total_recall = 0, 0
        len_total = len(convos_train)
        for i in range(len(convos_train)):
            question = convos_train[i][0]
            index = find_right_index_(question)
            answer = convos_train[index][1]
            good_answer = convos_train[i][1]
            precision, recall = precision_recall\
            (answer, good_answer)
            total_precision  += precision
            total_recall     += recall
        print('-'*80)
        total_precision = total_precision/len_total
        total_recall = total_recall/len_total     
        F = 2*total_precision*total_recall/(total_precision+total_recall)
        print('ON TRAINS, WE HAVE PRECISION = {:.5f},  RECALL = {:.5f}, F_MESURE = {:.3f} '\
              .format(total_precision, total_recall,F))
        

def return_F_score(): 
    
        total_precision, total_recall = 0, 0
        len_ = len(convos_test)
        for i in range(len(questions_test)):
            question = convos_test[i][0]
            index = find_right_index_(question)
            answer = convos_train[index][1]
            good_answer = convos_test[i][1]
            precision, recall = precision_recall\
            (answer, good_answer)
            total_precision  += precision
            total_recall     += recall
        total_precision = total_precision/len_
        total_recall = total_recall/len_     
        F = 2*total_precision*total_recall/(total_precision+total_recall)
        return F
 
    
def chatting_with_corpus():

    path = 'processed/chat_tfidf.txt'
    i = 0
    questions = []
    with open(path, 'r') as f:
        for line in f:
            if i%2 ==0:
                line = line[8:]
                line = line.strip().strip('-').strip('+')
                questions.append(line)
            i +=1
    f.close()
    good_index = []
    questions_ = []
    index = 0
    for question in questions:
        question = clear_line(question)
        if question not in questions_:
            questions_.append(question)
            good_index.append(index)
        index += 1
    questions_clean = [questions[i] for i in good_index]
    for line in questions_clean:
            print('Humaine: ', line)
            time, time_word = _find_right_time(line, COMPLET_DICT_TIME)
            list_city = _find_right_geography(line, CITY_LIST)
            if time_word is not None:
                line = line.replace(time_word, '')
            if list_city is not None:
                for city in list_city:
                    line = line.replace(city, '')
            LINE = clear_line(line)
            unknown_word =0
            for word in LINE.split():
                if word not in LIST_OF_WORDS:
                    unknown_word += 1
            if unknown_word > 1:
                print('Désolé, je ne comprends pas ta question, peux tu la reformuler s\'il te plaît?')                    
                print()
                continue
            if len(line) !=0:
                index= find_right_index_(LINE)
                if time == None and list_city == None:
                     print('Bot: ', convos_train[index][1])
                elif list_city ==None:
                    print('{}, {}'.format(time_word, convos_train[index][1]))
                elif time == None:
                    cities = ' '.join(list_city)
                    print('À {}, {}'.format(cities, convos_train[index][1]))
                else:
                    cities = ' '.join(list_city)
                    print('À {}, {}, {}'.format(cities, time_word, convos_train[index][1]))
            print()
               

    
                
def chatting_with_test_corpus():
    
    path = 'test.txt'
    questions = []
    with open(path, 'r') as f:
        for line in f:
                line = line.strip()
                questions.append(line)
    f.close()
    questions_clean = []
    for question in questions:
        if question not in questions_clean:
            questions_clean.append(question)
    for line in questions_clean:
            line1 = line
            time, time_word = _find_right_time(line, COMPLET_DICT_TIME)
            list_city = _find_right_geography(line, CITY_LIST)
            if time_word is not None:
                line = line.replace(time_word, '')
            if list_city is not None:
                for city in list_city:
                    line = line.replace(city, '')
            LINE = clear_line(line)
            unknown_word =0
            print('Vous: ', line1)
            for word in LINE.split():
                if word not in LIST_OF_WORDS:
                    unknown_word += 1
            if unknown_word > 1:
                print('Bot: Désolé, je ne comprends pas ta question, peux tu la reformuler s\'il te plaît?')
                print()
                continue
            if len(line) !=0:
                index= find_right_index(LINE, DICT, méthode)
                if time == None and list_city == None:
                     print('Bot: ', pairs_trains[index][1])
                elif list_city ==None:
                    print('Bot: {}, {}'.format(time_word, pairs_trains[index][1]))
                elif time == None:
                    cities = ' '.join(list_city)
                    print('Bot: À {}, {}'.format(cities, pairs_trains[index][1]))
                else:
                    cities = ' '.join(list_city)
                    print('Bot: À {}, {}, {}'.format(cities, time_word, pairs_trains[index][1]))
               
                LINE = clear_line(line)   
                index = find_right_index(LINE, DICT, METHODE[0])
                #print('YOU: ', line)
                #print('BOT: ', pairs_trains[index][1])
                print()
            

def get_all_and_all_convos():
    convos = []
    file1 = 'data/convos27juin_test.txt'
    file2 = 'data/convos27juin_train.txt'
    files = [file1, file2]
    for file in files:
        with open(file, 'r') as f:
            i=0
            for line in f:
                if i%2==0:
                    question = line.strip()
                    if '++++' in question:
                         question = question[9:]
                else:
                    answer = line.strip()
                    if '++++' in answer:
                        answer = answer[9:]
                    convos.append([question, answer])
                i+=1
        f.close()
    return convos


    
def dict_of_all_score_(data, KEY_LIST, sujet):
    """
    return score of all word in all doc"""
    DICT = {}
    for doc in data:
        DICT[doc]=dict_doc_score_(doc,KEY_LIST, sujet)
    return DICT 



def find_right_index_(new_doc, DICTIONARY):
    

    index=0
    biggest_score = score_by_new_doc_(new_doc, DATA[0], DICTIONARY)
    for i in range(len(DATA)):
        boolean1 = score_by_new_doc_(new_doc, DATA[i], DICTIONARY) > biggest_score
        boolean2 = score_by_new_doc_(new_doc, DATA[i], DICTIONARY) == biggest_score
        boolean3 = len(DATA[index]) >len(DATA[i])
        if boolean1 or (boolean2 and boolean3):
            biggest_score = score_by_new_doc_(new_doc, DATA[i], DICTIONARY)
            index = i
    if biggest_score == 0:
        index = random.choice([i for i in range(len(DATA))])
    return index



        



def request_missing_infos(new_question, good_question, dict_):
    
    if 'to_return' in good_question:
        return True
    score1 = score_by_new_doc_(new_question, good_question, dict_)
    score2 = 0
    for word in dict_[good_question]:
        score2 += dict_[good_question][word]
    if score1/score2 > 0.6:
        return  True
    else:
        return  False
    
def to_print_answer(line):
    line = str(line)
    first = line[0]
    line = first.upper()+ line[1:]
    if line[-1] not in ['?','.','!']:
        line = line+'.'
    line = line.strip(' ')
    return line
    
def chat_(): 
      
    DICTIONARY =  dict_for_all_doc(DATA)      
    print('Bonjour, le bot d\'Avicen à ton écoute')
    path = 'processed/chat_tfidf.txt'
    f =  open(path, 'a+') 
    while True:
            line = str(input('Vous: '))
            if len(line) < 2:
                break
            time, time_word = _find_right_time(line, COMPLET_DICT_TIME)
            list_city = _find_right_geography(line, CITY_LIST)
            if time_word is not None:
                line = line.replace(time_word, '')
            if list_city is not None:
                for city in list_city:
                    line = line.replace(city, '')
            LINE = data_processed.clear_line(line)
            unknown_word =0
            for word in LINE.split():
                if word not in LIST_OF_WORDS:
                    unknown_word += 1
            if unknown_word > 1:
                print('Désolé, je ne comprends pas ta question, peux tu la reformuler s\'il te plaît?')                    
                print()
                continue
            if len(line) !=0:
                index= find_right_index_(LINE, DICTIONARY)
                if time == None and list_city == None:
                     print('Bot: ', CONVOS_TRAIN[index][1])
                elif list_city ==None:
                    print('{}, {}'.format(time_word, CONVOS_TRAIN[index][1]))
                elif time == None:
                    cities = ' '.join(list_city)
                    print('À {}, {}'.format(cities, CONVOS_TRAIN[index][1]))
                else:
                    cities = ' '.join(list_city)
                    print('À {}, {}, {}'.format(cities, time_word, CONVOS_TRAIN[index][1]))
                
    
def dict_doc_score_(doc, KEY_LIST, sujet): 
    """
    We return score of all one words and all two consequent words"""

    list_word_ = data_processed.split_words(doc)    
    return {word: tf_idf_(word, doc, KEY_LIST, sujet) for word in list_word_}
   

    
def score_by_new_doc_(new_doc, doc, DICT):
    
    list_word_in_doc = data_processed.split_words(new_doc)
    score = 0
    for word in list_word_in_doc:
            if word in DICT[doc]:
                 score += DICT[doc][word]
    return score


def tf_idf_standard(word, doc):
    
    df = 0
    doc_split = data_processed.split_words(doc)
    if word in doc_split:
        tf = 1
    else:
        tf = 0
    for doc_ in DATA:
        if word in doc_:
            df+=1
    if df >0:
        return tf*math.log(len(DATA)/df, 10)
    else:
        return 0
    
def dict_of_entropy_for_all_terms(terms, K):
    """
    Nous utilisons la méthode donné dans Class Specific TF-IDF Boosting 2018
    K is a tuning parameter of method"""
    dictionnary_terms ={}
    H_max =0
    for term in terms:
        H = 0
        df = 0
        for data2 in DATA:
            if term in data2:
                df+=1
        for data in DATA:
            if term in data:
                tf_idf = - 1/df*math.log(1/df, 2)
            else:
                tf_idf = 0
            H += tf_idf
        if H > H_max:
            H_max = H
        dictionnary_terms[term] = H
    for term in terms:
        dictionnary_terms[term] = (H_max - dictionnary_terms[term])/(H_max*K)
    return dictionnary_terms

def dict_of_all_doc_by_new_method(K):
    
    """
    This will be the dictionnary which is the heart of method"""
    dict_of_new_method = {}
    terms = get_all_terms()
    dictionnary_terms = dict_of_entropy_for_all_terms(terms, K)
    for doc in DATA:
        terms_of_this_doc = split_words(doc)
        dict_term_for_this_doc = {term: tf_idf_standard(term, doc)+dictionnary_terms[term]\
                                  if tf_idf_standard(term, doc) >0 else 0\
                                  for term in terms_of_this_doc}
        dict_of_new_method[doc] = dict_term_for_this_doc
    return dict_of_new_method

def get_all_terms():
    """
    return all terms for training
    """
    all_terms = []
    for data in DATA:
        all_terms.extend(split_words(data))
    return all_terms
        
    

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
        

def convos_to_questions(convos):
    """
    prend tous les conversations dans corpus et renvoie un array de documents avec
    des vocabulaires séparés"""
    
    data = []
    for convo in convos:
        data.append(convo[0])
    return data

def get_answer(index):
    if index != None:
        return convos_train[index][1]
    else:
        return 'Désolé, je ne comprends pas ta question'
    
def list_words():
    list_ = []
    for data in DATA:
        for term in split_words(data):
            if term not in list_:
                list_.append(term)
    return list_


        