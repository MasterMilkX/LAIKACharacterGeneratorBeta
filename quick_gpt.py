## QUICK AND DIRTY GPT-MODEL CODE FOR GENERATING TEXT AND TRAINING ON CORPUSES
# Written by Milk


# NER
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import json

# train on corpus
from datasets import Dataset, load_dataset, load_metric, disable_progress_bar
from transformers import Trainer, TrainingArguments, default_data_collator
from itertools import chain

# imports for the models
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Tokenizer
)

import torch
import shutil
import os

#switch between gpu if avalaible or cpu
if torch.cuda.is_available():
	device = torch.device("cuda")   
else:
	device = torch.device("cpu")
    




class QuickGPT():
    def __init__(self,corpus=None,model_type='gpt2',direct_loc=None):
        # other variables to save
        self.cache_dir = "_cache_tmp"
        self.train_eval_split = 20
        self.model_dir = "corpus_models"
        
        
        
        #create the model and tokenizer outright
        if not corpus and not direct_loc:
            self.new_setup(model_type)
        else:
            self.import_model(corpus,model_type,direct_loc)

        


    #create a vanilla model and tokenizer
    def new_setup(self,mp='gpt2'):
        self.config = AutoConfig.from_pretrained(mp)
        self.config.use_cache = False
        #print(self.config)
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(mp)
        self.model = AutoModelForCausalLM.from_pretrained(mp, config=self.config)

        self.model.to(device)


    #import a saved model based on the corpus name (hardcoded directory)
    def import_model(self,corpus,model_type='gpt2',direct_loc=None):
        if direct_loc:
            self.model_out_loc = direct_loc 
        else:
            self.model_out_loc = f"{self.model_dir}/{corpus.split('/')[-1].replace('.txt','')}"
        
        self.config = AutoConfig.from_pretrained(model_type)
        self.config.use_cache = False
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_out_loc)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        self.model.to(device)

    #generate text from a prompt using the model
    def gen_txt(self,prompt,param=None):
        args = {
          "min_length":10,
          "max_new_tokens":50,
          "temperature":0.93,
          "repetition_penalty":1,
          "top_p":0.93,
          "num_return_sequences":5,
          "top_k":0,
          "do_sample":True,
          "remove_invalid_values":True
        }
        
        if param:
            for k,v in param.items():
                args[k] = v
        
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        output_ids = self.model.generate(prompt_ids,
                      #run_name=run_name,
                      **args
                      )
        responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return [response[len(prompt):] for response in responses]

   
    #### HELPER CODE FOR THE FINETUNE FUNCTION #####
    
    def encode(self,examples, tokenizer, text_column_name):
            return tokenizer(examples[text_column_name])
    
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(self,texts, block_size):
        # Concatenate all texts.
        concatenated_texts = {k: list(chain(*texts[k])) for k in texts.keys()}
        total_length = len(concatenated_texts[list(texts.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_texts.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    
    def compute_metrics(self,eval_preds):
        preds, labels = eval_preds
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return load_metric("accuracy").compute(predictions=preds, references=labels)


    def preprocess_logits_for_metrics(self,logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)
    
    
    
    # finetunes the model on a corpus text
    def finetune(self,corpus):
        # make directories
        corp_name = corpus.split('/')[-1].replace('.txt','')
        self.model_out_loc = f"{self.model_dir}/{corp_name}"
        #full_cache = os.path.join(self.cache_dir,corp_name)
        full_cache = os.path.join(self.model_out_loc,self.cache_dir)
        checkpoint_dir = os.path.join(self.model_out_loc,"checkpoints")
        
        #delete previous cache
        torch.cuda.empty_cache()
        try:
            shutil.rmtree(full_cache)
        except OSError as e:
            print("Error: %s : %s" % (full_cache, e.strerror))

        #### prepare the dataset for training   ####

        #setup data files
        data_files = {}
        data_files["train"] = corpus
        dataset: Dataset = load_dataset(
            "text", data_files=data_files, cache_dir=full_cache, keep_linebreaks=True
        )

        # split to validation and train
        dataset["validation"] = load_dataset(
            "text",
            data_files=data_files,
            cache_dir=full_cache,
            split=f"train[:{self.train_eval_split}%]",
        )
        dataset["train"] = load_dataset(
            "text",
            data_files=data_files,
            cache_dir=full_cache,
            split=f"train[{self.train_eval_split}%:]",
        )

        # get the columns(?)
        column_names = dataset["train"].column_names
        text_column_name: str = "text" if "text" in column_names else column_names[0]

        

        #run tokenizer on the dataset
        dataset = dataset.map(
            self.encode,
            batched=True,
            num_proc=1,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
            fn_kwargs={"tokenizer": self.tokenizer, "text_column_name": text_column_name},
        )

#         block_size: int = self.tokenizer.model_max_length
#         if block_size > 1024:
#             block_size = 1024
            
        block_size = 16
        MAX_STEPS = 600
#         MAX_STEPS = int(600 * (1024/block_size))


        lm_datasets: Dataset = dataset.map(
            self.group_texts,
            batched=True,
            num_proc=1,
            desc=f"Grouping texts in chunks of {block_size}",
            fn_kwargs={"block_size": block_size},
        )

        train_dataset: Dataset = lm_datasets["train"]
        eval_dataset: Dataset = lm_datasets["validation"]



        ### train the model  ###

        # get arguments (a lot)
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            overwrite_output_dir=True,
            save_steps=300,
            save_total_limit=1,
            evaluation_strategy="steps",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            seed=42069,
            optim="adamw_torch", 
#             num_train_epochs=10,
            max_steps=MAX_STEPS,
            resume_from_checkpoint=None
        )


        #setup trainer
        trainer = Trainer(
                model=self.model,
                train_dataset=train_dataset,
                args=training_args,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=default_data_collator,
                compute_metrics=self.compute_metrics,
                preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
            )

        # train i guess (try using GPU)
        tr_result = trainer.train(resume_from_checkpoint=None)

        # save the model
        print(f"**** EXPORTING MODEL TO {self.model_out_loc} ****")
        trainer.save_model(self.model_out_loc)
    
    
        shutil.rmtree(full_cache, ignore_errors=True)
#         shutil.rmtree(f"corpus_models/checkpoint-600", ignore_errors=True)
#         shutil.rmtree(f"corpus_models/checkpoint-*", ignore_errors=True)

