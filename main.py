import webscrape as web 
import chunks as ck
import embeddings as eb
import similarity as sm
import LoadLLM as llm

ET= web.ExtractText()
pages_and_texts=ET.urlDownloadLink(url="https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf")
print(pages_and_texts)
CK= ck.ChunksConversion()
pages_and_chunks_over_min_token_len=CK.Convert(pages_and_texts)
EB=eb.genrateEmbeddings("cuda")
EB.get(pages_and_chunks_over_min_token_len[0])
embeddings_df_save_path=EB.saveToCsv()
SM=sm.search("text_chunks_and_embeddings_df.csv","cuda")
LLM=llm.gemma("hf_token")
def setDialogue_template2(query:str,context_items:list[dict])->str:
        context="- "+"\n-".join([item["sentence_chunk"] for item in context_items])
        base_prompt= f"""
    
        take the help of context items and answer the query
        if the query does not match content don't answer 
        
        Context items :
        {context}
        Query:{query}"""
        
        print( base_prompt)
def Generate(query):
  scores_index_pages=SM.getEmbeddings(query)
  LLM.askGemma2(query,scores_index_pages[2],scores_index_pages[0],scores_index_pages[1])
  #setDialogue_template2(query,scores_index_pages[2])
  #print((scores_index_pages[2]))
Generate("who is Narendra Modi")
print("==============================================================")
#Generate("Who is Narendra Modi ")
print("==============================================================")
#Generate("Using Eyes of Discernment")
"""
LLM=llm.gemma("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")

while (True):
   data=input()
   if data!='END':
      
    LLM.askGemma1(data)
    continue
   break
"""