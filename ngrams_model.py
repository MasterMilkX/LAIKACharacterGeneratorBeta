from nltk.tokenize import word_tokenize
import random
import re

# N-Grams model for generating text based on a dataset
# modified from https://towardsdatascience.com/text-generation-using-n-gram-model-8d12d9802aa0
class NGramGen():
    def __init__(self,datafile,n=3):
        self.n = n
        self.create_ngram_set(datafile)
        
        
    # create an ngram dataset model with probabilities and what-not
    def create_ngram_set(self,datfile):
        gram_set = {}  #key = gram set (array), value = { ct = how many times the context shows, next_word = list of words occuring after and their counts}
        n = self.n
        
        # tokenize the data
        with open(datfile,"r") as f:
            for line in f:
                # break up the tokens by n-grams
                
                line2 = line.replace("<|startoftext|> ",("__START__ " * (n-1))).replace("<|endoftext|>","__END__")
                tokens = word_tokenize(line2)
                
                for i in range(len(tokens)-(n-1)):
                    g = tuple(tokens[i:i+n-1])   #get the gram
                    nw = tokens[i+n-1]         #get the next word
                    #print(f"{g} -> {nw}")
                    
                    #add to the gram set
                    if g not in gram_set:
                        gram_set[g] = {"ct":0,"next_word":{}}
                    gram_set[g]['ct']+=1
                    
                    #add count of seeing next word
                    if nw not in gram_set[g]['next_word']:
                        gram_set[g]['next_word'][nw] = 0
                    gram_set[g]['next_word'][nw]+=1
        self.gram_set = gram_set
      
    # return the bayesian probability of seeing a particular token given context
    def prob(self,context,token):
        tc = tuple(context)
        if tc in self.gram_set and token in self.gram_set[tc]['next_word']:
            return self.gram_set[tc]['next_word'][token] / self.gram_set[tc]["ct"]
        else:
            return 0.0
        
    # randomly select the next token given context
    def rand_token(self,context,auto_end=False):
        # assume context is in the gram set
        nd = self.gram_set[context]['next_word']
        words = []
        probs = []
        for w in nd.keys():
            words.append(w)
            probs.append(self.prob(context,w))
        if auto_end and "__END__" in words:
            return "__END__"
        #return random.choices(words,weights=probs,k=1)[0]   # weighted probability
        return random.choice(words)                          # normal probability
        
    # generate a sentence
    def generate(self,mintok=5):
        context = tuple(["__START__"] * (self.n-1))
        out_set = []
        next_token = self.rand_token(context)
        #next_token = "they"
        while next_token != "__END__":
            # add to output
            out_set.append(next_token)
            
            # extend context
            cl = list(context)
            cl.append(next_token)
            cl.pop(0)
            context = tuple(cl)
            
            # get the next context
            next_token = self.rand_token(context,(len(out_set) >= abs(mintok)))
        
        
        #return the combined text
        out_text = " ".join(out_set)
        out_text = re.sub(r'\s*(,|\.|\?)\s*', r'\1 ', out_text)
        out_text = re.sub(r'\s+(\“|\’)\s*',r'\1', out_text)
        out_text = re.sub(r'(\(|\))\s*',r'', out_text)
        
        return out_text
        