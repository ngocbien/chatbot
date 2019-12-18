def analyse_reponse(question, reponse):
    """
    arg: question is the question of user, reponse is given by chatbot
    C'est une fonction de con, juste pour tester
    """
    ids_vehicule = '29608'
    if '#' not in reponse:
        return reponse
    else:
        parameter = []
        if '#km' in reponse:
            temps = find_right_time(question)
            ids_vehicule = find_ids(question)
            if ids_vehicule == None:
                print( 'Précise quel véhicule tu parles')
            else:
                parameter.append(find_km(temps, ids_vehicule))
        if '#carburant#' in reponse:
            temps = find_right_time(question)
            ids_vehicule = find_ids(question)
            if ids_vehicule == None:
                print('Précise quel véhicule tu parles')
            else:
                parameter.append(find_carburant(temps, ids_vehicule))
        else: # "#conduite#" in reponse:
            temps = find_right_time(question)
            ids_vehicule = find_ids(question)
            if ids_vehicule == None:
                print( 'Précise quel véhicule tu parles')
            else:
                parameter.append(conduite(temps, ids_vehicule))
        return find_km('abc', ids_vehicule)       
       # if  '#trajet#' in reponse:
        #if ''
        
        
def find_right_time(temps):
    print('fonction en cours de construction')
    return 'con'
    
def find_ids(question):
    if 'tous' in question or 'toutes' in question:
        return 'all'
    else:
        return input('indiquez inditifiant de ce véhicule:')

def carburant(temps, ids_vehicule):
    
    #if ids_vehicule =='all':
    print('Fonction est en cours de construction')
    
def find_km(temps, id_vehicule):
    import pandas as pd
    df = pd.read_csv('fleet_donnees.csv', sep = ';')
    t = df.loc[(df['ID Véhicule'] == int(id_vehicule)) & (df['Debut période']=='22/01/2018')]
    return str(t['Distance'].values[0])+' km.'
    
        