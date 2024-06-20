import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available 
from huggingface_hub import login
from transformers import BitsAndBytesConfig

class gemma:
    quantization_config=""
    attn_implementation=""
    use_quantization_config=True
    model_id ="google/gemma-2b-it"
    tokenizer=""
    llm_model=""
   
    def __init__(self ,hf_token):
        login(token=hf_token)
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_compute_dtype=torch.float16)
        if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
            self.attn_implementation = "flash_attention_2"
        else:
            self.attn_implementation = "sdpa"
            print(f"[INFO] Using attention implementation: {self.attn_implementation}")
        print(f"[INFO] Using model_id: {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.model_id)
        self.llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.model_id, 
                                                 torch_dtype=torch.float16, # datatype to use, we want float16
                                                 quantization_config=self.quantization_config if self.use_quantization_config else None,
                                                 low_cpu_mem_usage=False, # use full memory 
                                                 attn_implementation=self.attn_implementation) # which attention version to use
    def setGemmaType(self):
        gpu_memory_bytes= torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb=round(gpu_memory_bytes/(2**30))
        print(gpu_memory_gb)
        if gpu_memory_gb < 5.1:
            print(f"Your available GPU memory is {gpu_memory_gb}GB, you may not have enough memory to run a Gemma LLM locally without quantization.")
        elif gpu_memory_gb < 8.1:
            print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in 4-bit precision.")
            self.use_quantization_config = True 
            self.model_id = "google/gemma-2b-it"
        elif gpu_memory_gb < 19.0:
            print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in float16 or Gemma 7B in 4-bit precision.")
            self.use_quantization_config = False 
            self.model_id = "google/gemma-2b-it"
        elif gpu_memory_gb > 19.0:
            print(f"GPU memory: {gpu_memory_gb} | Recommend model: Gemma 7B in 4-bit or float16 precision.")
            self.use_quantization_config = False 
            self.model_id = "google/gemma-7b-it"

        print(f"use_quantization_config set to: {self.use_quantization_config}")
        print(f"model_id set to: {self.model_id}")
    def sysGPUInfo(self):
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.current_device()}")
    def modelParamsInfo(self):
        def get_model_num_params(model: torch.nn.Module):
            return sum([param.numel() for param in model.parameters()])

        print(get_model_num_params(self.llm_model))
        def get_model_mem_size(model: torch.nn.Module):
            """
            Get how much memory a PyTorch model takes up.

            See: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822
            """
            # Get model parameters and buffer sizes
            mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
            mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])

            # Calculate various model sizes
            model_mem_bytes = mem_params + mem_buffers # in bytes
            model_mem_mb = model_mem_bytes / (1024**2) # in megabytes
            model_mem_gb = model_mem_bytes / (1024**3) # in gigabytes

            return {"model_mem_bytes": model_mem_bytes,
                    "model_mem_mb": round(model_mem_mb, 2),
                    "model_mem_gb": round(model_mem_gb, 2)}

        print(get_model_mem_size(self.llm_model))
    def setDialogue_template1(self,query):
        input_text=query
        dialogue_template =[
            {
                "role":"user",
                "content":input_text
            }
        ]
        prompt = self.tokenizer.apply_chat_template(conversation=dialogue_template,
                                            tokenize=False,
                                            add_generation_prompt=True)
        return prompt
    def setDialogue_template2(self,query:str,context_items:list[dict])->str:
        context="- "+"\n-".join([item["sentence_chunk"] for item in context_items])
        base_prompt= f"""
    
        take the help of context items and answer the query
        
        
        

        -------------
        Context items :
        {context}
        Query:{query}
        """
        
        return base_prompt
    def askGemma1(self , query):
        prompt=self.setDialogue_template1(query)
        input_ids=self.tokenizer(prompt,return_tensors="pt").to("cuda")
        outputs=self.llm_model.generate(**input_ids,max_new_tokens=500)
        #print(f"Model output token {outputs[0]}")
        outputs_decoded=self.tokenizer.decode(outputs[0])
        return outputs_decoded
    def askGemma2(self,query,pages_and_chunks,scores,indices):
        prompt=self.setDialogue_template2(query,pages_and_chunks)
        
        input_ids= self.tokenizer(prompt ,return_tensors="pt").to("cuda")

        outputs=self.llm_model.generate(**input_ids, temperature=0.7,
                                do_sample=True ,
                                max_new_tokens=500)

        output_text =self.tokenizer.decode (outputs[0])
        finaloutput=output_text.replace(prompt," ")
        l=["The context does not provide","false content","The context items do not provide","not provide","so I cannot answer"]
        for i in l:
            #print(i)
            if i in finaloutput:
                #print("checked")
                
                finaloutput = self.askGemma1(query)
                break
                
        print(f"\n\n\n\nQuery:{query} \n<-------------------------------------------->\n RAG answer \n{finaloutput}")