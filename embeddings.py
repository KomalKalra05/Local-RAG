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
from sentence_transformers import SentenceTransformer
from sentence_transformers import  util
class genrateEmbeddings:
    embedding_model=SentenceTransformer(model_name_or_path="all-mpnet-base-v2")
    pages_and_chunks_over_min_token_len2=[]
    text_chunks_embeddings=[]
    def __init__(self , Device ):
        #embedding_model=SentenceTransformer(model_name_or_path="all-mpnet-base-v2",device=Device)
        self.embedding_model.to(Device) # requires  GPU installed, for reference on my local machine, I'm using a NVIDIA RTX 4060
    def get(self ,pages_and_chunks_over_min_token_len):
        self.pages_and_chunks_over_min_token_len2=pages_and_chunks_over_min_token_len
        for item in tqdm(pages_and_chunks_over_min_token_len):
            item["embedding"] = self.embedding_model.encode(item["sentence_chunk"])
        text_chunks=[item["sentence_chunk"] for item in pages_and_chunks_over_min_token_len]
        print(text_chunks[420])
        text_chunks_embeddings=self.embedding_model.encode(text_chunks,batch_size=12,convert_to_tensor=True)
        print(text_chunks_embeddings)

    def saveToCsv(self):
        text_chunks_and_embeddings_df=pd.DataFrame(self.pages_and_chunks_over_min_token_len2)
        embeddings_df_save_path="text_chunks_and_embeddings_df.csv"
        text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index =False )
        return embeddings_df_save_path