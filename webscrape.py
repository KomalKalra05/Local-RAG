import os 
import requests
import fitz
from tqdm.auto import tqdm
import pandas as pd

class ExtractText:
    def __init__(self):
        print("intialized")
    def text_formatter(self,text:str)->str:
        cleaned_text=text.replace("\n"," ").strip()
        return cleaned_text
    def open_and_read_pdf(self,pdf_path:str)->list[dict]:
        doc =fitz.open(pdf_path)
        print(type(doc))
        pages_and_texts=[]
        for page_number , page in  tqdm (enumerate (doc)):
            text =page.get_text()
            text=self.text_formatter(text=text)
            pages_and_texts.append(
                {"page_number":page_number-41,
                "page_char_count":len(text),
                "page_word_count":len(text.split(" ")),
                "page_sentence_count_raw":len(text.split(". ")),
                "page_token_count":len(text)/4,
                "text":text}
            )
        return pages_and_texts
    def urlDownloadLink(self,url):
        dummy=url
        data=(dummy.split("//"))
        data=" ".join(data).split("/")
        data=data[1]
        pdf_path=data+".pdf"
        
        if not os.path.exists(pdf_path):
                print(f"not found ")
                filename=pdf_path
                response= requests.get(url)
                if response.status_code== 200:
                    with open(filename,"wb") as file :
                        file.write(response.content )
                    print(f"the file has be taken {filename }")
                    pages_and_texts=self.open_and_read_pdf(pdf_path=pdf_path)
                    return pages_and_texts
                else :
                    print(f"the file has be failed  {filename }")
                    return 
        else:
                print("file exists ")
                pages_and_texts=self.open_and_read_pdf(pdf_path=pdf_path)
                return pages_and_texts
        
"""
Extracttext=ExtractText()
print(Extracttext.urlDownloadLink(url="https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"))
"""