## QUICK AND DIRTY NAMED-ENTITY RECOGNITION CODE FOR EXTRACTING ENTITIES FROM A GIVEN TEXT
# Written by Milk


import spacy
import numpy as np
import random
import re
import os
import nltk
from nltk.corpus import stopwords
import requests
#from bs4 import BeautifulSoup
from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok

# get stop words
spacy_ner = spacy.load('en_core_web_trf')
stopW = set(stopwords.words("english"))

# get default names and places
local_dir = os.path.dirname(__file__)
default_values = {'names':[],'places':[]}
with open(os.path.join(local_dir,'names.txt'),'r') as nf:
    default_values['names'] = [x.strip() for x in nf.readlines()]
with open(os.path.join(local_dir,'places.txt'),'r') as pf:
    default_values['places'] = [x.strip() for x in pf.readlines()]


# extract all of the proper nouns from a text
def getProperNouns(text):
    dat = spacy_ner(text)
    prpns = []
    for token in dat:
        if token.pos_ == "PROPN":
            if token.text not in prpns:
                prpns.append(token.text)
    
    return prpns

# extract all of the verbs from a text
def getVerbs(text):
    dat = spacy_ner(text)
    verbs = []
    for token in dat:
        if token.pos_ == "VERB":
            if token.text not in verbs:
                verbs.append(token.text)
    return verbs

# extract all of the noun phrases from a text
def getNounPhrases(text):
    dat = spacy_ner(text)
    noun_phrases = [chunk.text for chunk in dat.noun_chunks]
    
    return noun_phrases

# extract all of the entities and their classes from a text
def getEntities(text):
    dat = spacy_ner(text)
    entis = []
    for entity in dat.ents:
        entis.append((entity.text, entity.label_))
    return entis

# extract all of the (possible) characters from a text
def getCharacters(text):
    e = getEntities(text)
    p = [x[0] for x in e if x[1] == "PERSON"]
    return np.unique(p).tolist()

# extract all of the (possible) places from a text
def getPlaces(text):
    e = getEntities(text)
    p = [x[0] for x in e if x[1] in ["FAC","GPE","ORG","GEO","LOC"]]
    return np.unique(p).tolist()
    

# removes all of the proper nouns and makes the text generic
def makeGenericText(text,replacer={'name':"X",'place':"Y"},debug=False):
    # get the proper nouns
    names = getCharacters(text)
    places = getPlaces(text)
    g_text = text[:]
    
    # replace real_name (and subsets referring to them (i.e. first or last name)
    i = 0
    rn = re.match(r"--\s(.+)\s--",text)
    all_real_names = []
    if rn:
        #grab the character's name
        real_name = rn.group(1)
        all_real_names = [real_name]
        for n in real_name.split(" "):
            if n == " " or n == "" or n in stopW:
                continue
            all_real_names.append(n)
            
        #replace all instances or subsets of the name
        for x in all_real_names:
            g_text = g_text.replace(x,f"#{replacer['name']}_{i}#")

        i = 1
    
    # replace all names in text
    for x in names:
        if re.match(r"-- (.+) --",x) or x in all_real_names:
            continue
        g_text = g_text.replace(x,f"#{replacer['name']}_{i}#")
        i+=1
        
    # replace all places in text
    i = 0
    for y in places:
        g_text = g_text.replace(y,f"#{replacer['place']}_{i}#")
        i+=1   
        
    if(debug):
        print(f"REPLACED: {all_real_names,names,places}")
        
    return g_text


# replace parts of the generic text with the prompt's nouns
def replaceGeneric(genTxt, prompt, replacer={'name':"X",'place':"Y"},debug=False):
    # if duplicate pronouns, remove
    genTxt = re.sub(f"(#{replacer['name']}_([0-9]+)#\s?)+",r"\1",genTxt)
    genTxt = re.sub(f"(#{replacer['place']}_([0-9]+)#\s?)+",r"\1",genTxt)
    
    # get the proper nouns
    names = getCharacters(prompt)
    places = getPlaces(prompt)
    
    #get the name of the character (or a random character if not given)
    real_name_s = re.findall(r"--\s(.+)\s--",prompt,re.MULTILINE)
    real_name = real_name_s[0] if real_name_s else random.choice(names)
    
    #filter the other proper names
    other_names = [x for x in names if x not in real_name and "--" not in x]
    places = [x for x in places]  #idk it keeps erroring out -_-
    
    #randomize order
    random.shuffle(names)
    random.shuffle(places)
        
    #get all generic names
    name_nums = np.unique([int(x) for x in re.findall(f"#{replacer['name']}_([0-9]+)#",genTxt)])
    place_nums = np.unique([int(y) for y in re.findall(f"#{replacer['place']}_([0-9]+)#",genTxt)])
    max_name = 0 if len(name_nums) == 0 else max(name_nums)+1
    max_place = 0 if len(place_nums) == 0 else max(place_nums)+1
    
    #fill till enough
    while len(other_names) < max_name:
        other_names.append(random.choice(default_values['names']))
    while len(places) < max_place:
        places.append(random.choice(default_values['places']))
        
        
    #replace all real name instances
    fillTxt = str(genTxt)
    fillTxt = re.sub(f"#{replacer['name']}_0#",real_name,fillTxt)
    
    if debug:
        print(f"[1-{max_name}] -> {other_names}")
        print(f"[0-{max_place}] -> {places}")
       
    #replace other names
    for ni in range(1,max_name):
        fillTxt = re.sub(f"#{replacer['name']}_{ni}#",other_names[ni-1],fillTxt)
        
    #replace all places
    for pi in range(max_place):
        fillTxt = re.sub(f"#{replacer['place']}_{pi}#",places[pi],fillTxt)
        
    return fillTxt


# replace the selected character from the a blurb text if they show up again
def replaceMC(prompt,blurb,MC=None):
    # get the proper nouns
    names = getCharacters(prompt)
    
    #get the name of the character (or a random character if not given)
    if not MC:
        real_name_s = re.findall(r"--\s(.+)\s--",prompt,re.MULTILINE)
        real_name = real_name_s[0] if real_name_s else random.choice(names)
    else:
        real_name = MC
    
    #filter the other proper names
    other_names = [x for x in names if x not in real_name and "--" not in x]
    if len(other_names) == 0:
        other_names = random.choices(default_values['names'],k=5)
        
   
    #replace MC name
    b2 = blurb[:]
    while real_name in b2:
        b2 = b2.replace(real_name, random.choice(other_names),1)
    return b2


#replace all possessive MC's with another character to be safe
def replacePossessive(prompt,txt,MC=None):
    # get the proper nouns
    names = getCharacters(prompt)
    
    #get the name of the character (or a random character if not given)
    if not MC:
        real_name_s = re.findall(r"--\s(.+)\s--",prompt,re.MULTILINE)
        real_name = real_name_s[0] if real_name_s else random.choice(names)
    else:
        real_name = MC
        
    #filter the other proper names
    other_names = [x for x in names if x != real_name and "--" not in x]
    if len(other_names) == 0:
        other_names = random.choices(default_values['names'],k=5)
        
    
    #replace MC name
    txt2 = txt[:]
    while f"{real_name.strip()}’s" in txt2: 
        txt2 = txt2.replace(f"{real_name.strip()}’s", f"{random.choice(other_names).strip()}’s")
    return txt2


# returns the dictionary synonyms of a given word
def synonyms(word):
    #scrape from https://dictionaryapi.dev/
    response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
    if response.status_code != 200:
        print("ERROR: Could not receive data from API :/")
        return []
    jr = json.loads(json.dumps(response.json()))
    
    #parse the synonyms
    syn_list = []
    for j in jr:
        for m in j['meanings']:
            syn_list += m['synonyms']
    return syn_list


    # (old code) Scrape synonyms from thesaurus.com
    # response = requests.get('https://www.thesaurus.com/browse/{}'.format(term))
    # soup = BeautifulSoup(response.text, 'html.parser')
    # soup.find('section', {'class': 'css-191l5o0-ClassicContentCard e1qo4u830'})
    # return [span.text.strip() for span in soup.findAll('a', {'class': 'css-1kg1yv8 eh475bn0'})]


def replaceSynonym(txt):
    txt_sent,pos_sent, _ = textPosPair(txt)
    
    #get all of the adjectives
    adj_list = []
    for i in range(len(pos_sent)):
        if pos_sent[i] == "ADJ":
            adj_list.append(txt_sent[i])
    
    # if there are duplicates, replace them in the original text
    for a in adj_list:
        d = txt_sent.count(a)
        if d > 1:
            ai = [x for x in range(len(txt_sent)) if txt_sent[x] == a]
            syn_list = synonyms(a)
            if len(syn_list) < len(ai)-1:  #can't find enough synonyms...
                continue
            
            syns = random.choices(syn_list,k=len(ai)-1)
            for i in range(1,len(ai)):
                txt_sent[ai[i]] = syns[i-1]
    #retokenize
    return retok(txt_sent)
            
# retokenizes a sentence back together
def retok(tokens):
    detokenizer = Detok()
    text = detokenizer.detokenize(tokens)
    text = re.sub(r'\s*(,|\.|\?)\s*', r'\1 ', text)
    return text


# add post-processing effects to the generated descriptions
# - replace all occurences of main character's name
# - replace duplicate adjectives with synonyms
# - replace the main character possessive with another character
def descPostProc(prompt,gen_txt,mc=None):
    #new_gen_txt = replaceMC(prompt,gen_txt,mc)
    new_gen_txt = replaceSynonym(gen_txt)
    new_gen_txt = replacePossessive(prompt,new_gen_txt,mc)
    return new_gen_txt


# get the text and POS pairing of a text
def textPosPair(text):
    # get the tokens and their text-POS pairing
    dat = spacy_ner(text)
    text_sent = []
    pos_sent = []
    tag_sent = []
    for token in dat:
        #skip the character genericism
        if token.text == "#":
            continue
        text_sent.append(token.text)
        pos_sent.append(token.pos_)
        tag_sent.append(token.tag_)
    return text_sent, pos_sent, tag_sent

# show pos of a blurb of text
def debugTextPos(text):
    t,p,e = textPosPair(text)
    s = []
    for i in range(len(t)):
        s.append(f"{t[i]}({p[i]} - {e[i]})")
    print(" ".join(s))
    print(p)
    
