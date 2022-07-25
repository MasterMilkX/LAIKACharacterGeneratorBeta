## GENERATES CHARACTER SUMMARIES IN A MAD-LIBS STYLE BY COMPARING LISTS OF WORD SETS BASED ON PARTS-OF-SPEECH AND COMPARES TO PROMPT PROXIMITY TO RANDOMLY CONSTRUCTING BLURBS
# Written by Milk

import numpy as np
import random
import re
import os

#for encoder model
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

# uses lists of word types and grammars to generate quick character descriptions
class MadLibsTraits():
    def __init__(self,wordTypeSet=None,gram=None):
        
        #use default filepaths for each set if not set
        if not wordTypeSet:
            wordTypeSet = {
                        'NOUN':'character_corpuses/POS-nouns.txt',
                        'PNOUN':'character_corpuses/POS-job_nouns.txt',
                        'ADJ':'character_corpuses/POS-adjectives.txt',
                        'PADJ':'character_corpuses/POS-personal_adj.txt',
                        'VERB':'character_corpuses/POS-verb.txt',
                        'ANOUN':'character_corpuses/POS-abstract_nouns.txt',
                        'EVERB':'character_corpuses/POS-emo_verb.txt'
                          }
            
        

        #set the grammars (default or passed)
        if not gram:
            self.grammars = [
                ["a","<PADJ>","<PNOUN>"],
                ["a","<PADJ>","<ADJ>","<PNOUN>"],
                ["a","<PADJ>","<ADJ>","<PNOUN>","and","<ADJ>","<PNOUN>"],
                ["a","<PADJ>","<PNOUN>","<VERB>ing", "<NOUN>s"],
                ["a","<PADJ>","<PNOUN>","who","<VERB>s","<NOUN>s"],
                ["a","<PADJ>","<PNOUN>","who","<EVERB>s","<NOUN>s"],
                ["a","<PADJ>","<PNOUN>","that","<EVERB>s","<ANOUN>s"]
            ]
        else:
            self.grammars = gram
        
        #import the model
        self.model = SentenceTransformer('all-mpnet-base-v2')                 #MPNET (best performance from exp - see Holy Grail exp)
        print("> imported sentence encoder model!")
        
        #import data for each word type
        #need at least adj, noun, and verb
        self.wt = {}
        for pos,f in wordTypeSet.items():
            print(f"> importing and encoding: {pos}")
            self.wt[pos] = {"words":[],'encode':[]}
            self.wt[pos]['words'] = self.getFileDat(f)
            self.wt[pos]['encode'] = self.encodeWords(self.wt[pos]['words'])
            
        
    #import raw line data
    def getFileDat(self, file):
        dat = []
        local_dir = os.path.dirname(__file__)
        with open(os.path.join(local_dir,file), 'r') as f:
            dat = f.readlines()
            dat = [l.strip() for l in dat]
        return dat
    
    #encode the words using the model for use later
    def encodeWords(self,ws):
        enc_ws = self.model.encode(ws)
        return enc_ws
    
    # make some "character summaries" from the grammars (n=# of blurbs, s=# closest from each group)
    def charSum(self,prompt,n=3,s=7,debug=False):
        #encode prompt and get the closest of the sets
        prompt_enc = self.model.encode([prompt])
        
        close_words = {}
        for k in self.wt:
            w_enc_set = self.wt[k]['encode']
            dist = np.array(cos_sim(prompt_enc,w_enc_set)[0])
            close_ind = np.argsort(dist)[::-1][:s].flatten()  #get top 5 closest indexes
            close_words[k] = {}
            close_words[k]['words'] = [self.wt[k]['words'][i] for i in close_ind]
            close_words[k]['dists'] = [dist[i] for i in close_ind]
            
            #calculate probability to use a specific word based on the distance division
            t = sum(close_words[k]['dists'])
            close_words[k]['probs'] = [(x/t) for x in close_words[k]['dists']]
            
        if debug:
            for k,v in close_words.items():
                print(f"-- {k} --")
                for i in range(len(v['words'])):
                    print(f"{v['words'][i]}: {v['dists'][i]:.2f} -> {v['probs'][i]:.2f}")
                print("")

        #make the blurbs from the grammars and the proximity set
        b = []
        for ni in range(n):
            b.append(self.makeBlurbs(close_words))
            
        return b
    
    #check if word starts/ends with a vowel
    def vow_start(self,w):
        return w[0] in ['a','e','i','o','u']
    def vow_end(self,w):
        return w[-1] in ['a','e','i','o','u']
    
    #make blurbs using the closest sets
    def makeBlurbs(self,close_set):
        g = random.choice(self.grammars)
        
        #match to the random grammar structure
        gfit = []
        chosen = []
        for i in range(len(g)):
            tok = g[i]
            
            #add as normal
            if "<" not in tok:    
                gfit.append(tok)
                
            #add randomly selected
            else:
                pos = re.sub('[<>sing]','',tok)
                sel = random.choices(close_set[pos]['words'],weights=close_set[pos]['probs'],k=1)[0]
                ci = 0
                while sel in chosen and ci < 5:
                    sel = random.choices(close_set[pos]['words'],weights=close_set[pos]['probs'],k=1)[0]
                    ci += 1
                add_gi = sel[:]
                chosen.append(sel)
                
                #### MODIFIERS ####
                
                #pluralize noun or make singular form verb
                if ">s" in tok:
                    if add_gi[-1] in ["h", "s"]:
                        add_gi += "e"
                    add_gi += "s"
                
                #custom verb modifiers
                if "VERB" in pos:
                    if "ing" in tok:
                        if self.vow_end(add_gi):
                            add_gi = add_gi[:-1] + "ing"
                        else:
                            add_gi += 'ing'
                        
                #turn to "an" if vowel in first letter
                if i > 0 and gfit[i-1] == "a" and self.vow_start(add_gi):
                    gfit[i-1] = "an"
                
                #add to the blurb
                gfit.append(add_gi)
                
        #combine together
        return " ".join(gfit)
    
    # get the best traits based on proximity to the prompt sets (copied from holy grail version but modified)
    def bestTraits(self,user_prompt,descs,bestNum=3):
        all_trait = []
        for de in descs:
            all_trait += self.charSum(de,n=int(bestNum*1.5))
            
        # get the closest based on the original prompt
        upro_enc = self.model.encode([user_prompt])
        trait_enc = self.model.encode(all_trait)
        t_dist = cos_sim(upro_enc,trait_enc)[0]
        
        close_ind = np.argsort(t_dist)[::-1][:bestNum].flatten()
        best_t = [all_trait[i] for i in close_ind]
        worst_t = [t for t in all_trait if t not in best_t]
        
           
        return best_t, worst_t
