# ============================= Define Dataset Class ============================= #
# handle text preprocessing and tokenization (converting text into tensors that BERT can process)

import config
import torch

class BERTDataset:
    def __init__(self, reviews, targets):
        self.reviews = reviews
        self.targets = targets # 0 or 1
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    # return the length of the dataset
    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        # process each review individually
        review = str(self.reviews[item])
        # remove weird spaces
        review = " ".join(review.split())
        # convert HTML newlines to spaces
        review = review.replace("<br />", " ")

        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True, # Add [CLS] and [SEP]
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        padding_length = self.max_len - ids.size(1)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[item], dtype=torch.float).unsqueeze(0) # unsqueeze to add an extra dimension
        }