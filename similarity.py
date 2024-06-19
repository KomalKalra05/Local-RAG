import torch 
import numpy as np 
import pandas as pd 
import textwrap
from sentence_transformers import util,SentenceTransformer
class search:
    device=""
    embeddings_df_save_path=""
    pages_and_chunks=[]
    def __init__(self,embeddings_df_save_path,device):
        self.device=device
        self.embeddings_df_save_path=embeddings_df_save_path
        
        print(self.pages_and_chunks)
    def toEmbeddings(self):
        text_chunks_and_embeddings_df=pd.read_csv("text_chunks_and_embeddings_df.csv")
        #converting embedding column to np.array
        text_chunks_and_embeddings_df["embedding"]=text_chunks_and_embeddings_df["embedding"].apply(lambda x:np.fromstring(x.strip("[]") ,sep=" "))
        #convert our embeddings into a torch.tensor 
        embeddings=torch.tensor(np.stack(text_chunks_and_embeddings_df["embedding"].tolist(),axis=0),dtype=torch.float32).to("cuda")
        self.pages_and_chunks=text_chunks_and_embeddings_df.to_dict(orient="records")
        print(text_chunks_and_embeddings_df)
        print(embeddings.shape)
        return embeddings
    def useModel(self):
        embedding_model=SentenceTransformer(model_name_or_path="all-mpnet-base-v2",device=self.device)
        return embedding_model
    def print_wrapped(self,text):
        wrapped_text=textwrap.fill(text,80)#wrap_length =80
        #print(wrapped_text) 
        return wrapped_text
    def retrieve_relevant_resources(self,query:str,embeddings:torch.tensor,
                              
                                n_resources_to_return:int= 5,
                                print_time:bool=True):
        model=self.useModel()
        query_embedding=model.encode(query,convert_to_tensor=True).to("cuda")
        #start_time=timer()
        dot_scores=util.cos_sim(a=query_embedding, b=embeddings)[0]
        #dot_scores=util.dot_score(a=query_embedding, b=embeddings)[0]
        #end_time=timer()
        #print(f"{end_time-start_time}")
        scores,indices =torch.topk(input=dot_scores,k=n_resources_to_return)

        return scores, indices
    def print_top_results_and_scores(self,query:str, embeddings:torch.tensor,n_resources_to_return: int =5):
        scores,indices=self.retrieve_relevant_resources(query=query,
                                                embeddings=embeddings,
                                                n_resources_to_return=n_resources_to_return)
        for score , idx in zip(scores,indices):
            print(f"Score {score}")
            print("Text:")
            self.print_wrapped(self.pages_and_chunks[idx]["sentence_chunk"])
            print(f"Page no {self.pages_and_chunks[idx]["page_number"]}")
            print("\n")
    def print_top_results_and_scores(self,scores , indices):
        for score , idx in zip(scores,indices):
            print(f"Score {score}")
            print("Text:")
            print(self.pages_and_chunks[idx]["sentence_chunk"])
            print(f"Page no {self.pages_and_chunks[idx]["page_number"]}")
            print("\n")
    
    def getEmbeddings(self,query):
        embeddings=self.toEmbeddings()
       
        scores,indices=self.retrieve_relevant_resources(query=query,
                                                embeddings=embeddings,
                                                 
                                                n_resources_to_return=5)
        #self.print_top_results_and_scores(scores,indices)
        #print(self.pages_and_chunks[indices[0]+49])
        final_pages=[]
        for score , idx in zip(scores,indices):
           
            final_pages.append(self.pages_and_chunks[idx])
            
        
        return [scores, indices,final_pages]