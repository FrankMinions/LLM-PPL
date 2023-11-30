import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

device = torch.device('cuda:0')


class Arguments:
    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path


class LLM:
    def __init__(self, configs):
        self.tokenizer = LlamaTokenizer.from_pretrained(configs.model_name_or_path, trust_remote_code=True)
        self.model = LlamaForCausalLM.from_pretrained(configs.model_name_or_path, trust_remote_code=True, device_map="auto")


if __name__ == '__main__':
    configs = Arguments(model_name_or_path="")
    llm = LLM(configs)

    corpus = '''def fibonacci(n):
                    if n <= 1:
                        return n
                    else:
                        return fibonacci(n-1) + fibonacci(n-2)'''

    inputs = llm.tokenizer(corpus, return_tensors="pt")
    loss = llm.model(input_ids=inputs["input_ids"].to(device), labels=inputs["input_ids"].to(device)).loss
    ppl = torch.exp(loss).detach().item()
    print(f"ppl={ppl}")
