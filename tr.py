import requests
import re
import torch
import gc
from transformers import AutoModelForCausalLM,AutoTokenizer
def prot(tg,n=5):
    a,b,c=[],'',0
    for t in tg:
        t=t.replace('[]','')
        b+=t+'[]'
        c+=1
        if c>=n:
            b=b[:-2]
            a.append(b)
            b=''
            c=0
    if b:
        a.append(b[:-2])
    return a
def det(tg):
    a=[]
    for t in tg:
        a.extend(t.split('[]'))
    return a

def cl(text):
    k=re.compile(r'\{\\K\d+\}')
    text=k.sub('',text)
    return text
def tr(tg,ct='',tl='中文',ot2tt="",mp='',tr_choice="transformers"):
    ttg=[]
    if tr_choice=="ollama":
        tg=prot(tg)
        for t in tg:
            t=cl(t)
            prompt=f'''{ct}
            参考上面的信息，参考下面的翻译：
            {ot2tt}
            
            将以下文本翻译为{tl}，注意只需要输出翻译后的结果，不要额外解释，保留原文中的'[]'。：
            {t}'''
            p= {
                "model":mp,
                "prompt":prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1  # 低温度，翻译更稳定
                }
            }
            retry_count = 0
            while retry_count <5:
                try:
                    r=requests.post("http://localhost:11434/api/generate", json=p)
                    r.raise_for_status()
                    result=r.json()
                    tr= result['response'].strip()
                    ttg.append(tr)
                    break
                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    # 重试前释放显存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    gc.collect()
                    if retry_count >= 5:
                        retry_count += 1
        ttg=det(ttg)
    elif tr_choice=="transformers":
        a=""
        for t in tg:
            a+=f"{t}\n"
        a=cl(a)
        tokenizer = AutoTokenizer.from_pretrained(mp)
        model = AutoModelForCausalLM.from_pretrained(mp, dtype="auto", device_map="cuda")
        messages = [{"role": "system","content": f"请贴合语义的下列内容翻译成{tl}，要求符合原文语境，不要添加额外的注释与内容。"},
                    {"role": "user", "content":a.strip("\n")}]
        text = tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        input_token_count = model_inputs.input_ids.shape[1]
        print("input_token_count:", input_token_count)
        generated_ids = model.generate(**model_inputs, max_new_tokens=262144)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        output_token_count = len(output_ids)
        total_token_count = input_token_count + output_token_count
        print("output_token_count:", output_token_count, "\ntotal_token_count:", total_token_count)
        content = tokenizer.decode(output_ids, skip_special_tokens=True).split("</think>")[1].strip("\n")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        ttg=(content.split("\n"))
    return ttg
def trsg(t,ct='',tl='中文',ot2tt="",mp='',tr_choice="transformers"):
    t = cl(t)
    if tr_choice=="ollama":
        prompt = f'''{ct}
                参考上面的信息，参考下面的翻译：
                {ot2tt}
    
                将以下文本翻译为{tl}，注意只需要输出翻译后的结果，不要额外解释。：
                {t}'''
        p = {"model": mp,"prompt": prompt,"stream": False,"options": {"temperature": 0.1}}
        retry_count = 0
        while retry_count < 5:
            try:
                r = requests.post("http://localhost:11434/api/generate", json=p)
                r.raise_for_status()
                result = r.json()
                tr = result['response'].strip()
                return tr
            except requests.exceptions.RequestException as e:
                retry_count += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                gc.collect()
                if retry_count >= 5:
                    print("retry")
                    retry_count += 1
        return "出错"
    elif tr_choice=="transformers":
        tokenizer = AutoTokenizer.from_pretrained(mp)
        model = AutoModelForCausalLM.from_pretrained(mp, dtype="auto", device_map="auto")
        messages = [{"role": "system","content": f"请贴合语义的下列内容翻译成{tl}，要求符合原文语境，不要添加额外的注释与内容。"},
                    {"role": "user", "content":t}]
        text = tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        input_token_count = model_inputs.input_ids.shape[1]
        print("input_token_count:", input_token_count)
        generated_ids = model.generate(**model_inputs, max_new_tokens=262144)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        output_token_count = len(output_ids)
        total_token_count = input_token_count + output_token_count
        print("output_token_count:", output_token_count, "\ntotal_token_count:", total_token_count)
        content = tokenizer.decode(output_ids, skip_special_tokens=True).split("</think>")[1].strip("\n")
        return content
if __name__ == '__main__':
    ttg=tr([])
    print(ttg)