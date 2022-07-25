## SELECTS TRAITS EXTRACTED FROM THE HOLY GRAIL WEBSITE TO USE WITH SENTENCE ENCODING AND INPUT CHARACTER PROMPTS RELEVANCY
# Written by Milk

import blurb_gen as BGen

import numpy as np
from tqdm import tqdm
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

class HolyGrailTraits():
    
    #setup for prompt and selection
    def __init__(self,datasets=['ALL'],mod_name='mpnet'):
        
        #list of segmented files
        self.TRAIN_FILES_DAT = {
            'astro':{'file':'character_corpuses/character_descriptions_astrology_CLEAN.txt',"bullet":"⭐️"},
            'adj':{'file':'character_corpuses/character_adjectives.txt',"bullet":"🍎"},
            'dark_mbti':{'file':'character_corpuses/character_descriptions_dark_mbti.txt',"bullet":"😈"},
            'hexaco':{'file':'character_corpuses/character_descriptions_hexaco.txt', "bullet":"⬣"},
            'mbti':{'file':'character_corpuses/character_descriptions_mbti.txt', "bullet":"😄"},
            'tt_flaws':{'file':'character_corpuses/character_descriptions_tvtrope_flaws.txt', "bullet":"📺"},
            'wattpad':{'file':'character_corpuses/character_descriptions_wattpad500.txt','bullet':"📝"}
        }
        
        #import encoding model
        model_names = {'mpnet':'all-mpnet-base-v2','bert':'distilbert-base-nli-mean-tokens','qa':'multi-qa-MiniLM-L6-cos-v1'}
        self.enc_model = SentenceTransformer(model_names[mod_name])                 
        print("> imported sentence encoder model!")
        
        #setup data to train on
        t_dat = []
        for k in self.TRAIN_FILES_DAT.keys():
            if 'ALL' in datasets or k in datasets:
                tdf = {}
                tdf['id'] = k
                tdf['file'] = self.TRAIN_FILES_DAT[k]['file']
                t_dat.append(tdf)
        
        #import train data
        self.importDat(t_dat)
            
            
    #import traits from the file and encode them with the model
    def importDat(self,TRAIN_DAT):
        self.full_dat = {}
        with tqdm(total=len(TRAIN_DAT)) as pbar:
            for tfd in TRAIN_DAT:
                fi = {}
                #get raw traits
                tok_traits = []
                local_dir = os.path.dirname(__file__)
                with open(os.path.join(local_dir,tfd['file']),'r') as f:
                    dat = f.readlines()
                    fi['traits'] = np.unique([BGen.regexFixer(d.strip().lower()) for d in dat])
                    
                #encode the traits
                fi['enc_traits'] = self.enc_model.encode(fi['traits'])
                
                #save
                self.full_dat[tfd['id']] = fi
                pbar.update(1)
        print(f"> imported and encoded traits for {list(self.full_dat.keys())}!")
        
        
        
    #returns indexes of traits from the set the are closest to the comparison text's encoding
    def topPicks(self,ctxt, enc_set, k=3):
        ctxt_enc = self.enc_model.encode([ctxt])
        dist = []
        for be in enc_set:
            d = cos_sim(ctxt_enc,[be])[0][0]
            dist.append(d)
        topK = np.argsort(dist)[::-1][:k]
        return topK, np.array(dist)[topK]
        
    #returns a list of top picked traits based on a prompt
    def getTraits(self,txt,max_traits=3,datasets=['ALL']):
        if 'ALL' in datasets:
            datasets = list(self.full_dat.keys())
        
        trait_str_set = []
        trait_dist_set = []
        best_of_group = []
        for d in datasets:
            #get selection for dataset
            if d not in self.full_dat:
                print(f"ERROR! Dataset [{d}] not imported! Cannot select traits from this dataset!")
                continue
            dat = self.full_dat[d]
            b = self.TRAIN_FILES_DAT[d]['bullet']
            ind,d = self.topPicks(txt,dat['enc_traits'],max_traits)

            #save top k traits as strings with bullet points
            for i in range(max_traits):
                s = f"{b} {dat['traits'][ind[i]]}"
                trait_str_set.append(s)
                trait_dist_set.append(d[i])
                if i == 0:
                    best_of_group.append(s)
                
        return trait_str_set, trait_dist_set, best_of_group
    
    # get the best traits based on proximity to the prompt sets
    def bestTraits(self,user_prompt,descs,bestNum=3):
        BEST_TRAIT_NUM = bestNum
        all_traits = []
        pair_traits = []
        best_of = {}
        
        for de in descs:
            treats,dists,best = self.getTraits(de)
            
            #save all of the best
            for b in best:
                bdi = dists[treats.index(b)]
                if b[0] not in best_of or bdi > best_of[b[0]]['dist']:
                    best_of[b[0]] = {'dist':bdi,'txt':b}
            
#             for i in range(len(treats)):
#                 t = treats[i]
#                 di = dists[i]
                
#                 #get distance of copy and replace with closer distance
#                 if t in all_traits:
#                     tind = [x[0] for x in pair_traits].index(t)
#                     if pair_traits[tind][1] < di:
#                         pair_traits[tind] = (t,di)
#                 else:
#                     all_traits.append(t)
#                     pair_traits.append((t,di))

            #compare selected trait to the original prompt
            # get the closest based on the original prompt
            # upro_enc = self.enc_model.encode([user_prompt])
            # trait_enc = self.enc_model.encode(treats)
            # t_dist = cos_sim(upro_enc,trait_enc)[0]

            # close_ind = np.argsort(t_dist)[::-1][:int(len(t_dist)/2)].flatten()#[:bestNum].flatten()
            # for i in close_ind:
            #     pair_traits.append((treats[i],t_dist[i]))
                
            
        #sort by closest distance
        # sort_traits = [y[0] for y in sorted(pair_traits, key = lambda x: x[1])]
        # best_traits = sort_traits[:bestNum]
        
        #add best from group
        best_traits += [x['txt'] for x in best_of.values()]
        best_traits = np.unique(best_traits)
        
        # bad_traits = [x for x in sort_traits if x not in best_traits]
        # bad_traits = np.unique(bad_traits)
        #bad_traits = bad_traits[:int(len(best_traits)*1.5)]
        
        best_traits.sort()
        # bad_traits.sort()
            
        
        return best_traits#, bad_traits
        