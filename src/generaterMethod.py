import torch, re
from tqdm import tqdm as tqdm
from datasets import Dataset
from datasets.utils import disable_progress_bar
from itertools import combinations
from transformers import DefaultDataCollator

class generater:
    def __init__(self, model, tokenizer, gen_model, gen_tokenizer, batch_size = 64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.gen_model = gen_model
        self.gen_tokenizer = gen_tokenizer
        self.batch_size = batch_size

    def tokenizeWithoutLabel(self, data):
        max_length = 128
        tokenized_datas = self.tokenizer(
            data['texts'],
            max_length=max_length,
            padding="max_length",
            truncation="only_second"
        )
        return tokenized_datas

    def deleteListIndex(self, tokens, indexs):
        out = tokens[:]
        indexs = list(indexs)
        for index in indexs[::-1]:
            if out[index] == '[UNK]':
                continue
            del out[index]
        return out

    def deleteStyleToken(self, text, batch_size):
        disable_progress_bar()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        texts = []
        texts.append({'ids': [0], 'texts' : text})
        tokens = self.tokenizer.encode(text)
        token_indexs = range(1, len(tokens) - 1)
        for n in range(1,4):
            if len(token_indexs) < n - 1:
                continue
            for indexs in combinations(token_indexs, n):
                texts.append({'ids' : indexs, 'texts' : self.tokenizer.decode(self.deleteListIndex(tokens, indexs)[1:-1])})
        line_data = Dataset.from_list(texts)
        line_tokenized_datasets = line_data.map(self.tokenizeWithoutLabel, batched=True, remove_columns=line_data.column_names)
        data_collator = DefaultDataCollator(return_tensors="pt")
        line_loader = torch.utils.data.DataLoader(line_tokenized_datasets, batch_size=batch_size,
                                                shuffle=False, collate_fn=data_collator,
                                                num_workers=0)
        max_diff = 0
        max_info = {}
        vanil_prob = 0
        for up_i, id_inputs in enumerate(line_loader):
            inputs = {}
            inputs['input_ids'] = id_inputs['input_ids'].to(self.device)
            inputs['token_type_ids'] = id_inputs['token_type_ids'].to(self.device)
            inputs['attention_mask'] = id_inputs['attention_mask'].to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits
            softmax = logits.softmax(dim=-1)
            for i, prob in enumerate(softmax):
                if up_i == 0 and i == 0:
                    if prob[1] < 0.7:
                        return {'masked':text, 'original':text}
                    else:
                        vanil_prob = prob
                else:
                    if vanil_prob[1] - prob[1] > max_diff:
                        max_info = (texts[up_i*batch_size + i], prob)
                        max_diff = vanil_prob[1] - prob[1]
        if max_diff >= 0.2:
            return {'masked':max_info[0]['texts'], 'original':text}
            
    def tokenizeMasked(self, data):
        text = self.deleteStyleToken(data, self.batch_size)
        if text['masked'] == text['original']:
            return
        tokenized_datas = self.gen_tokenizer(
            f"<unused0> <unused1> {text['masked']} <unused2>",
            return_tensors="pt"
        )
        return tokenized_datas

    def makeMoralText(self, immoral_text):
        input = self.tokenizeMasked(immoral_text)
        if not input:
            return immoral_text
        gen_ids = self.gen_model.generate(**input,
                                max_length=256,
                                pad_token_id=self.gen_tokenizer.pad_token_id,
                                eos_token_id=self.gen_tokenizer.eos_token_id,
                                bos_token_id=self.gen_tokenizer.bos_token_id)
        output = self.gen_tokenizer.decode(gen_ids[0])
        pred = re.search('2\>\s(.+?)\s\<u', output)
        ans=pred.group(1)
        return ans