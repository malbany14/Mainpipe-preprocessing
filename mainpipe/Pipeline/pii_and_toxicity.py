import presidio
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import pandas as pd
from pipeline import PipelineStep
import re

def flag_toxic_keywords(text):
    """
    Flag based on list of dirty naughty obscene  and otherwise bad words (english)
    https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words
    """
    try:
        with open('../../data/raw/en.txt', 'r', encoding='utf-8') as f:
            bad_words = [line.strip() for line in f if line.strip()]

        pattern = re.compile(r'\b(' + '|'.join(map(re.escape, bad_words)) + r')\b', flags=re.IGNORECASE)
        return bool(pattern.search(text))
    except Exception as e:
        print("ERROR in flag_toxic_keywords:", e)
        return False

def mask_text(text):
    phone = re.compile(r'\b(?:\+?61|0)[2-478](?:[ -]?\d){8}\b')
    email = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
    tfn = re.compile(r'\b\d{3}\s?\d{3}\s?\d{3}\b')

    masked_items = []

    if re.search(email, text):
        text = re.sub(email, "[EMAIL_MASKED]", text)
        masked_items.append("email")

    if re.search(phone, text):
        text = re.sub(phone, "[PHONE_MASKED]", text)
        masked_items.append("phone")

    if re.search(tfn, text):
        text = re.sub(tfn, "[TFN_MASKED]", text)
        masked_items.append("tfn")

    return text, ", ".join(masked_items) if masked_items else None

class PiiRemovalStep(PipelineStep):
    def __init__(self, name, validator=None):
        super().__init__(name, validator)

    def run(self, df):
        df[['text', 'masked_items']] = df['text'].apply(lambda x: pd.Series(mask_text(x)))
        return df

class ToxicRemovalStep(PipelineStep):
    def __init__(self, name, validator=None):
        super().__init__(name, validator)
        self.removed_rows = pd.DataFrame()    # Store rows removed due to toxicity

    def run(self, df):
        """
        Remove rows based on keyword filtering
        """

        df['is_inappropriate'] = df['text'].apply(flag_toxic_keywords)

        self.removed_rows = df[df["is_inappropriate"] == True] # set toxic rows in self.removed_rows

        df = df[df["is_inappropriate"] == False]
        return df