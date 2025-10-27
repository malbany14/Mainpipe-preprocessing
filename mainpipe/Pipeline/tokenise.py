from transformers import AutoTokenizer
from pipeline import PipelineStep
import pandas as pd

class TokenizationStep(PipelineStep):
    def __init__(self, name, validator=None, model_name="gpt2", max_length=512, batch_size=1000):
        super().__init__(name, validator)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length
        self.batch_size = batch_size

    def run(self, df):
        texts = df["text"].tolist()
        all_token_ids = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            tokenized = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np"
            )
            all_token_ids.extend(tokenized["input_ids"])

        df["token_ids"] = all_token_ids
        df["token_ids"] = [ids.tolist() if hasattr(ids, "tolist") else ids for ids in all_token_ids]
        return df
