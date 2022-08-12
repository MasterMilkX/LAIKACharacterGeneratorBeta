## BLURB GENERATION, POST-PROCESSING, EVALUATION, GROUPING, AND OTHER FUNCTIONS
# Written by Milk

#try pairwise similarity with tf-idf and cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import numpy as np

import re

import quick_ner as NER

import warnings
warnings.filterwarnings(action = 'ignore')

stop_words = set(stopwords.words('english'))

#import the encoding model
MODEL = SentenceTransformer('all-mpnet-base-v2')

# Turn a paragraph of information into bullet point text based on a POS grammar format


# find index of list in another list 
# helper function for 'bulletPointText()'
def list_index(big, small):
    all_m = []
    for i in range(len(big)-len(small)+1):
        for j in range(len(small)):
            if small[j] != "*" and big[i+j] != small[j]:
                break
        else:
            all_m.append(i)
    return all_m

# merges overlapping intervals in a list
# from https://www.geeksforgeeks.org/merging-intervals/ because i failed comp. theory twice lol
def mergeIntervals(arr):
    arr.sort(key = lambda x: x[0])
    m = []
    s = -10000
    maxInt = -100000
    for i in range(len(arr)):
        a = arr[i]
        if a[0] > maxInt:
            if i != 0:
                m.append((s,maxInt))
            maxInt = a[1]
            s = a[0]
        else:
            if a[1] >= maxInt:
                maxInt = a[1]
    if maxInt != -100000 and (s, maxInt) not in m:
        m.append((s, maxInt))
    return m



# turns text into bullet pointed info blurbs based on a grammar and the parts of speech of the sentence
def bulletPointText(text,post_proc=True,debug=False,custom_gram=None):
    # list of grammars to use for extracting blurbs of text
    if not custom_gram:
        GRAMMARS = [
            ["(VERB)","( PART)","(.*?)","( NOUN)"],
            ["(VERB)","(.*?)","( CCONJ)?","( NOUN)"],
            ["(VERB)","(.*?)","( NOUN)","(.*?)","(( NOUN)|( PROPN))+"],
            ["(AUX)","( DET)","(.*?)","(( NOUN)|( PROPN))+"],
            ["(AUX)","(.*?)","(( NOUN)|( PROPN))+"],
            ["(AUX)","(.*?)","(( NOUN)|( PROPN)|( VERB))"],
            ["(AUX)","( DET)","(.*?)","(( NOUN)|( PROPN))+"],
            ["(DET)","( NOUN)","(.*?)","(( NOUN)|( PROPN))+"],
            ["(DET)","( ADJ)","( PUNCT)?","( ADJ)?","( NOUN)"],
            ["(DET)","( ADV)?","( ADJ)?","( NOUN)","( ADP)","( ADV)?","( ADJ)?","( PROPN)?","(( NOUN)|( PROPN))"],
            ["(NOUN)","( ADP)","(.*?)","(( NOUN)|( PROPN))+"],
            ["(PROPN)","( PART)","( NOUN)"],
            ["(PRON)", "( NOUN)", "( AUX)", "(( ADV)|( ADJ))+"]
        ]
    else:
        GRAMMARS = custom_gram
        
    g_txt_match = {}
    for gi in range(len(GRAMMARS)):
        g_txt_match[gi] = []
        
    text_matches = []
    pos_parts = []
    tag_parts = []
    
    #split by sentence (just in case)
    sentences = text.split(".")
    for st in sentences:
    
        #get the text-pos pairing 
        text_sent, pos_sent, tag_sent = NER.textPosPair(st)


        # search for every grammar type in the text using REGEX
        pos_ind_set = []
        for gi in range(len(GRAMMARS)):
            g = GRAMMARS[gi]

            #combine into sentences for easier searching
            join_pos_sent = " ".join(pos_sent)
            join_grammar = "".join(g)

            # get the matches using regex
            matches = re.finditer(join_grammar,join_pos_sent)
                
            # convert matches back to text
            for m in matches:
                # get positions
                si = m.start(0)
                ei = m.end(0)
                
                # check if word positions have already been used
                pi = (si,ei)
                if pi in pos_ind_set:
                    continue
                else:
                    pos_ind_set.append(pi)
                    
                # for the debug later
                if debug:
                    # get number of spaces
                    space_start = join_pos_sent[:si].count(" ")
                    space_end = join_pos_sent[:ei].count(" ")

                    #get the text from the number of spaces (1 pos : 1 text token)
                    full_txt = " ".join(text_sent[space_start:space_end+1])
                    g_txt_match[gi].append(full_txt)
                    
                    
        # merge overlapping position intervals together
        all_pos_ind_set = mergeIntervals(pos_ind_set)
        
        # get the blurbs from the overlaps
        for p in all_pos_ind_set:
            # get number of spaces
            space_start = join_pos_sent[:p[0]].count(" ")
            space_end = join_pos_sent[:p[1]].count(" ")

            #get the text from the number of spaces (1 pos : 1 text token)
            full_txt = " ".join(text_sent[space_start:space_end+1])
            text_matches.append(full_txt)
            
            if post_proc:
                # get the pos and tag part (for post-processing)
                full_pos = pos_sent[space_start:space_end+1]
                pos_parts.append(full_pos)

                full_tag = tag_sent[space_start:space_end+1]
                tag_parts.append(full_tag)
                    
    if debug:
        for k,v in g_txt_match.items():
            print(f"# matches for grammar #{k}: {len(v)}")
            print(v)
            
            
    # apply generifier and singularizer for blurbs
    if post_proc:
        
        #custom set
        conj_v = {'were':'was', 'are':'is', "have":"has", "be":"is"}
#        conj_v = {'was':'were', 'is':'are', "has":'have'}
        remover_set = ["WP$"]
        pronoun1 = ["he","she","his","her","him","her"] # -> they
        
        #process each blurb
        for i in range(len(text_matches)):
            if tag_parts[i][0] == "VBP" or tag_parts[i][0] == "VB":   #found plural verb
                
                tmspl = text_matches[i].split(" ")
                v1 = tmspl[0]
                
                if debug:
                    print(v1)
                
                # convert 'to be' verbs
                if v1 in conj_v:
                    tmspl[0] = conj_v[v1]
                    
                #convert all else based on the last few characters
                else:
                    if v1[-1] == 'y':
                        tmspl[0] = v1[:-1] + 'ies'
                    elif v1[:-2] in ['ch','sh']:
                        tmspl[0] = v1[:-2] + 'es'
                    elif v1[-1] in ['o','s','x','z']:
                        tmspl[0] = v1[:-1] + 'es'
                    else:
                        tmspl[0] = v1 + 's'
                    
                #put back together
                #print(f"{text_matches[i]} -> {' '.join(tmspl)}")
                text_matches[i] = " ".join(tmspl)
                
            #remove starter words
            if tag_parts[i][0] in remover_set:
                tmspl = text_matches[i].split(" ")
                text_matches[i] = " ".join(tmspl[1:])
                
        
            # remove gendered nouns
            is_a = re.search(r"\b(is|was)\b .*?\b(a|an)\b", text_matches[i])
            if is_a:
                # delete the "a" or "an" and the first noun (most likely the gendered subject) 
                tmspl = text_matches[i].split(" ")
                a_index = tmspl.index("a") if "a" in tmspl else tmspl.index("an")
                
                #print(pos_parts[i])
                
                if "NOUN" in pos_parts[i][a_index:]:
                    noun_index = pos_parts[i][a_index:].index("NOUN")
                    noun = tmspl[a_index:][noun_index]
                    
                    #print(noun)

                    #remove the gendered noun
                    if noun in ["man","woman","girl","boy"]:
                        for a in [tmspl,pos_parts[i],tag_parts[i]]:
                            #print(a)
                            a.pop(noun_index+a_index)
                            #print(a)
                            a.pop(a_index)
                            #print(a)

                        #add it back
                        #print(f"{text_matches[i]} -> {' '.join(tmspl)}")
                        text_matches[i] = " ".join(tmspl)
                
            #apply regex fixes to remove gendered nouns and other broken punctuation
            text_matches[i] = regexFixer(text_matches[i])
            
            
              
    return text_matches


#applies regex fixes to a text
def regexFixer(txt):
    txt_copy = txt[:]
    #remove pronouns (the regex fae stealing your gender lol)
    #txt_copy = re.sub(r"\bhe\b|\bshe\b","they",txt_copy)
    txt_copy = re.sub(r"\bhis\b|\bher\b","their",txt_copy)
    #txt_copy = re.sub(r"\bhim\b|\bher\b","them",txt_copy)
    txt_copy = re.sub(r"\bhim\b |\bher\b ","",txt_copy)
    txt_copy = re.sub(r"\bhimself\b |\bherself\b ","themself",txt_copy)
    txt_copy = re.sub(r"\bhe\b |\bshe\b ","they",txt_copy)


    #quick and old gender noun remover
    txt_copy = re.sub(r"\bman\b|\bwoman\b","person",txt_copy)
    txt_copy = re.sub(r"\bmen\b|\bwomen\b","people",txt_copy)
    txt_copy = re.sub(r"\bboy\b|\bgirl\b","child",txt_copy)
    txt_copy = re.sub(r"\bboys\b|\bgirls\b","children",txt_copy)
    txt_copy = re.sub(r"\bson\b|\bdaughter\b","child",txt_copy)
    txt_copy = re.sub(r"\bgrandson\b|\bgranddaughter\b","grandchild",txt_copy)
    txt_copy = re.sub(r"\bgrandfather\b|\bgrandmother\b","grandparent",txt_copy)
    txt_copy = re.sub(r"\bmother\b|\bfather\b","parent",txt_copy)
    txt_copy = re.sub(r"\bsister\b|\bbrother\b","sibling",txt_copy)
    txt_copy = re.sub(r"\bhusband\b|\bwife\b","spouse",txt_copy)


    #combine possessive punctations
    txt_copy = re.sub(r" \'","'",txt_copy)
    txt_copy = re.sub(r" \’","'",txt_copy)

    txt_copy = re.sub(r" n\'t","n't",txt_copy)
    txt_copy = re.sub(r" n\’t","n't",txt_copy)

    #fix broken possisive punctation
    txt_copy = re.sub(r"^(\’s)","is",txt_copy)

    # conbine comma punctuation
    txt_copy = re.sub(r" \,",",",txt_copy)

    return txt_copy



# GROUP SIMILAR BLURBS TOGETHER BASED ON VECTOR WORD ENCODING


# group the blurbs based on the distance matrix (key = index tuple, value = dist)
def cluster_dist_mat(dm,max_i,thresh=None,num_groups=None):
    if thresh == None and num_groups == None:
        print("ERROR! Must specify max number of groups or distance threshold value")
        return None
    
    # sort the distances (desc)
    pairs = list(dm.keys())
    distances = list(dm.values())
    sort_dist_ind = np.argsort(np.array(distances))[::-1]
    
    # make clusters
    clusters = list(range(max_i))
    
    for di in sort_dist_ind:
        d = distances[di]
        
        # outside threshold, end
        if thresh != None and d < thresh:
            break
        #check cluster num, end
        elif num_groups != None and len(np.unique(clusters)) <= num_groups:
            break
            
        #otherwise, regroup - turn all minority cluster group labels to the majority
        p = pairs[di]
        p0_c = clusters[p[0]]
        p1_c = clusters[p[1]]
        
        majority_label = p1_c if clusters.count(p0_c) < clusters.count(p1_c) else p0_c
        minority_label = p1_c if majority_label == p0_c else p0_c
        
        for i,c in enumerate(clusters):
            if c == minority_label:
                clusters[i] = majority_label
        
    # combine into lists
    blurb_groups = []
    for cg in np.unique(clusters):
        blurb_groups.append(np.where(clusters == cg)[0])
    return blurb_groups



# group a list of text blurbs together using Word2Vec encoding
def group_blurbs_W2V(blurbs,thresh=None,num_groups=None,debug=False):

    vec_blurbs = [vectorize(b) for b in blurbs]
    
    #get cosine distances from average of sets of word vectors
    dist_mat = {}
    for x in range(0,len(blurbs)):
        for y in range(x,len(blurbs)):
            if x != y:
                x_vec = np.expand_dims(np.array(vec_blurbs[x]),axis=0)
                y_vec = np.expand_dims(np.array(vec_blurbs[y]),axis=0)
                cos_dist = cosine_similarity(x_vec,y_vec)[0][0]
                dist_mat[(x,y)] = cos_dist
                
    if debug:
        for k,v in dist_mat.items():
            if v >= thresh:
                print(f"{blurbs[k[0]]} - {blurbs[k[1]]} => {v}")
        print("")
                
    # combine indexes back to blurb groups
    gb = cluster_dist_mat(dist_mat,len(blurbs),thresh=thresh,num_groups=num_groups)
    blurb_clust = []
    for l in gb:
        blurb_clust.append(np.array(blurbs)[l].tolist())
    return sorted(blurb_clust,key=lambda x:len(x),reverse=True)
    
    
             
def vectorize(txt):
    return MODEL.encode(txt)
    

def strike(text):
    result = ''
    for c in text:
        result = result + c + '\u0336'
    return result
    

    
# retrieves the most relevant blurbs from a set (using distance in vector space)
def getBestBlurbs(prompt,blurb_set,close=2,far=1,debug=False):

    prompt_vec = vectorize(prompt)
    
    bset_vec = []
    for bs in blurb_set:
        svec = []
        for b in bs:
            svec.append(vectorize(b))
        bset_vec.append(svec)
        
    # reduce each blurb group to one representative
    best_blurbs = []
    best_bvecs = []
    for bsi in range(len(bset_vec)):
        # take the blurb that is closest to the prompt
        dists = [cosine_similarity([bv],[prompt_vec])[0][0] for bv in bset_vec[bsi]]
        max_dist = max(dists)
        best_ind = dists.index(max_dist)

        # take the longest blurb
#         longest_blurb = max(blurb_set[bsi],key=lambda x: len(x))
#         best_ind = blurb_set[bsi].index(longest_blurb)
        
        best_bvecs.append(bset_vec[bsi][best_ind])
        best_blurbs.append(blurb_set[bsi][best_ind])
        
    if debug:
        print(f"Best: {best_blurbs}")
        
        
    # find the top X,Y blurb vectors CLOSEST and FURTHEST to the prompt vector
    blurb_dists = [cosine_similarity([bv],[prompt_vec])[0][0] for bv in best_bvecs]
    #blurb_dists = [np.linalg.norm(bv-prompt_vec) for bv in best_bvecs]
    blurb_dists = [blurb_dists[i]*len(best_blurbs) for i in range(len(best_blurbs))]
    selected_index = []
    farthest = np.argsort(np.array(blurb_dists))
    closest = farthest[::-1]
    
    if debug:
        print(f"Closest ({[blurb_dists[i] for i in closest[:(close+2)]]}): {[best_blurbs[i] for i in closest[:(close+2)]]}")
        print(f"Farthest ({[blurb_dists[i] for i in farthest[:(far+2)]]}): {[best_blurbs[i] for i in farthest[:(far+2)]]}")
        print("")
    
    # get the indices of the top X and Y of each
    ct = 0
    for x in closest:
        if ct == close:
            break
        if x not in selected_index:
            selected_index.append(x)
            ct += 1
        
    ct = 0
    for y in farthest:
        if ct == far:
            break
        if y not in selected_index:
            selected_index.append(y)
            ct += 1
        
            
    # get the leftovers that didn't make the cut in order of selection proximity
    selected_blurbs = [best_blurbs[i] for i in selected_index]
    removed_blurbs = []
    flat_blurbs = [x for xs in blurb_set for x in xs]
    for fb in flat_blurbs:
        if fb not in selected_blurbs:
            removed_blurbs.append(fb)
    
    return selected_blurbs, removed_blurbs