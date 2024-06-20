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
def Generate(query):
  scores_index_pages=SM.getEmbeddings(query)
  LLM.askGemma2(query,scores_index_pages[2],scores_index_pages[0],scores_index_pages[1])
  #setDialogue_template2(query,scores_index_pages[2])
  #print((scores_index_pages[2]))
Generate("who is Narendra Modi")
print("==============================================================")
Generate("What are macronutrients")
print("==============================================================")