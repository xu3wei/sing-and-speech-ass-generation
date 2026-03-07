from transformers import AutoModelForCausalLM, AutoTokenizer
import re
def cl(text):
    k=re.compile(r'\{\\K\d+\}')
    text=k.sub('',text)
    return text
def prot(tg):
    a,b,c=[],'',[]
    for t in tg:
        t.replace('|','')
        a.append(t)
        if len(a)>=10:
            b="|end|".join(a)
            a=[]
            c.append(b)
    if a:
        b="|end|".join(a)
        c.append(b)
    return c
def det(t):
    a=[]
    a.extend(t.replace('|结束|','|end|').replace('|end>','|end|').replace('<end|','|end|').split('|end|'))
    return a
def tr(tg,ct='',tl='中文',ot2tt='',mp='HY-MT1.5-7B'):
    ttg=[]
    tokenizer = AutoTokenizer.from_pretrained(mp)
    model = AutoModelForCausalLM.from_pretrained(mp, device_map="cuda",dtype="bfloat16")  # You may want to use bfloat16 and/or move to GPU here
    tg=prot(tg)
    for t in tg:
        t=cl(t)
        prompt=f'''{ct}
        参考上面的信息，参考下面的翻译：
        {ot2tt}
        
        将以下日语文本翻译为{tl}，注意只需要输出翻译后的结果，不要额外解释,其中|end|为格式信息，请保留：
        {t}'''
        messages = [{"role": "user", "content":prompt}]
        tokenized_chat = tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,return_tensors="pt")
        outputs = model.generate(tokenized_chat.to(model.device), max_new_tokens=2048,top_k=20,top_p=0.6,repetition_penalty=1.05,
        temperature=0.7,do_sample=True,num_return_sequences=1,pad_token_id=tokenizer.eos_token_id)
        output_text = tokenizer.decode(outputs[0]).split("<|extra_0|>")[1][:-8]
        output_text=det(output_text)
        if len(output_text)==10:
            ttg.extend(output_text)
        elif len(output_text)<10:
            x=10-len(output_text)
            ttg.extend(output_text)
            ttg+=['可能窜行']*x
        else:
            ttg.extend(output_text[:10])
    return ttg
def trsg(t,ct='',tl='中文',ot2tt='',mp='HY-MT1.5-7B'):
    tokenizer=AutoTokenizer.from_pretrained(mp)
    model=AutoModelForCausalLM.from_pretrained(mp, device_map="cuda", dtype="bfloat16")
    t=cl(t)
    prompt = f'''{ct}
           参考上面的信息，参考下面的翻译：
           {ot2tt}

           将以下日语文本翻译为{tl}，注意只需要输出翻译后的结果，不要额外解释,其中|end|为格式信息，请保留：
           {t}'''
    messages = [{"role": "user", "content": prompt}]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,return_tensors="pt")
    outputs = model.generate(tokenized_chat.to(model.device), max_new_tokens=2048, top_k=20, top_p=0.6,repetition_penalty=1.05,
                             temperature=0.7, do_sample=True, num_return_sequences=1,pad_token_id=tokenizer.eos_token_id)
    output_text = tokenizer.decode(outputs[0]).split("<|extra_0|>")[1][:-8]
    return output_text