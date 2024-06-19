import os 
import requests
import fitz
from tqdm.auto import tqdm
import pandas as pd 
#new1
import re 
import pandas as pd 
from spacy.lang.en import English
nlp=English()
nlp.add_pipe("sentencizer")
class ChunksConversion:
    pages_and_chunks=[]
    def __init__ (self):
        print("chunks intiated ")
    def Convert(self ,pages_and_texts):
        df = pd.DataFrame(pages_and_texts)
        for item in tqdm(pages_and_texts):
            item["sentences"]= list (nlp(item["text"]).sents)
            # safe side making everything thing into a string 
            item["sentences"]=[str(i) for i in item["sentences"]]

            #flagging how many sentences are there 
            item["page_sentence_count_spacy"]= len(item["sentences"])

        num_sentence_chunk_size=10
        def split_list(input_list:list[str],slice_size:int=num_sentence_chunk_size)->list[list[str]]:
            return [input_list[i:i+slice_size] for i in range(0,len(input_list),slice_size)]

        #print(pages_and_texts[300])
        #df = pd.DataFrame(pages_and_texts)

        for item in tqdm(pages_and_texts):
            item["sentences_chunks"]= split_list(input_list=item["sentences"], slice_size=num_sentence_chunk_size)
            item["num_chunks"]=len(item["sentences_chunks"])

     
        for item in tqdm (pages_and_texts):
            for sentence_chunk in item["sentences_chunks"]:
                chunk_dict={}
                chunk_dict["page_number"]=item["page_number"]

                joined_sentence_chunk="".join(sentence_chunk).replace("  "," ").strip()
                joined_sentence_chunk=re.sub(r'\.([A-Z])',r'.\1',joined_sentence_chunk)

                chunk_dict["sentence_chunk"]=joined_sentence_chunk
                chunk_dict["chunk_char_count"]=len(joined_sentence_chunk)
                chunk_dict["chunk_word_count"]=len([word for word in joined_sentence_chunk.split(" ")])
                chunk_dict["chunk_token_count"]=len(joined_sentence_chunk)/4  
                #1 token =4char

                self.pages_and_chunks.append(chunk_dict)
        print(len(self.pages_and_chunks))
        #print(self.pages_and_chunks)
        df =pd.DataFrame(self.pages_and_chunks)
        print(df)
        min_token_length=30
        pages_and_chunks_over_min_token_len=df[df["chunk_token_count"]>min_token_length].to_dict(orient="records")
        return [pages_and_chunks_over_min_token_len,self.pages_and_chunks]