import re

def formatting_func_standard(ds, eos_token=None):

    if eos_token is None:                              # in evaluation step
          eos_token = ''
    ds_lst = ds['text']
    for idx, string in enumerate(ds_lst):
        formatted_text = re.sub('### Assistant','### assistant',string) + eos_token
        ds_lst[idx] = formatted_text

    return {'text':ds_lst}

def formatting_func_chat(ds):
            
            ds_lst = ds['text']
            for idx, string in enumerate(ds_lst):
                formatted_text = re.sub('### Assistant:','[/INST]',string)
                formatted_text = re.sub('### assistant:','[/INST]',formatted_text)
                formatted_text = re.sub('### Human','### human',formatted_text)
                formatted_text = '[/INST]'.join(formatted_text.split('### human:')[1:])
                system_prompt = '''A helpful assistant, who is an expert in the fields of philosophy and literary studies, takes questions and tasks from a human. The assistant provides responses that appropriately complete the request in the same language that the human used.'''
                chat_text = f'''[INST] <<SYS>>{
                    system_prompt}
                    <</SYS>> 
                    {formatted_text}</s>'''
                ds_lst[idx] = chat_text

            return {'text':ds_lst}