import pandas as pd
from pipeline import PipelineStep
from langdetect.lang_detect_exception import LangDetectException
from langdetect import detect
import trafilatura
import ftfy
import re
from deduplication import split_paragraphs
from collections import Counter

def repetitiveness_score(text, n=3):
    """
    Split text to n-grams, count duplicates and divided by total n-gram count
    """

    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    counts = Counter(ngrams)
    total = len(ngrams)
    repeated = sum(v for v in counts.values() if v > 1)
    return repeated / total

def detect_language(text):
    """
    Use langdetect to return the language of text and unknown if no language
    """
    try:
        return detect(text)
    except LangDetectException:
        return "Unknown"
    
def clean_html_trafilatura(text):
    """
    Using trafilatura library clean html elements
    """
    extracted = trafilatura.extract(text)
    return extracted if extracted else text

def clean_special_characters(text: str) -> str:
    """
    Function to clean special characters from text and normalise whitespace
    """
    text = re.sub(r'[âÂÃ¢€‹„”¢¦§¨©ª«¬­®¯°±²³´µ¶·¸¹º»¼½¾¿]', '', text)
    #text = re.sub(r'\s+', ' ', text).strip()
    # REMOVED whitespace stripping as we need paragraphs for fuzzy deduplication
    #Remove general symbol ranges (e.g., currency, dingbats, box drawings)
    text= re.sub(r'[\u20A0-\u20CF\u2100-\u214F\u2190-\u21FF\u2500-\u257F\u2580-\u259F]', '', text)
    return text

class NullCleaningStep(PipelineStep):
    def __init__(self, name:str, validator):
        super().__init__(name, validator)
    
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out NA's
        """
        self.removed_rows = df[df['text'].isna()]
        df = df[df['text'].notna()]
        return df

class UTF8EncodingStep(PipelineStep):
    def __init__(self, name:str, validator):
        super().__init__(name, validator)
    
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix and ensure UTF-8 encoding for text
        """
        df['text'] = (
            df['text']
            .fillna('')
            .astype(str)
            .apply(lambda x: ftfy.fix_text(x))
        )
        return df

class SpecialCharacterCleaningStep(PipelineStep):
    def __init__(self, name, validator):
        super().__init__(name, validator)
    
    def run(self,df: pd.DataFrame) ->pd.DataFrame:
        """
        After the UTF8 encoding is fixed, some noisy special characters remain, remove these
        """
        df['text'] = df['text'].apply(clean_special_characters)
        return df

class LanugageCleaningStep(PipelineStep):
    def __init__(self, name, validator):
        super().__init__(name, validator)

    def run(self, df):
        """
        Detect languages and filter out non-english text
        """
        df['language'] = df['text'].apply(detect_language)

        # store removed rows
        self.removed_rows = df[df['language'] != 'en']

        df = df[df['language'] == 'en']

        return df
    
class HtmlCleaningStep(PipelineStep):
    def __init__(self, name, validator):
        super().__init__(name, validator)
    
    def run(self, df):
        """
        Use trafilatura on a row by row basis to clean html elements
        """
        df["text"] = df["text"].apply(clean_html_trafilatura)
        return df
    
class CaseNormalisationStep(PipelineStep):
    def __init__(self, name, validator=None):
        super().__init__(name, validator)
    
    def run(self, df):
        """
        Convert all text to lowercase
        """
        df["text"] = df["text"].str.lower()
        return df
    
class QualityFilteringSTep(PipelineStep):
    def __init__(self, name, validator=None):
        super().__init__(name, validator)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform various quality filtering tests on text data
        """
        # remove document outliers in word count
        # to do: convert to a function
        df["word_count"] = df["text"].apply(lambda x: len(x.split()))
        removed_rows = df[~((df["word_count"]>30) & (df["word_count"]<15000))]
        self.removed_rows = removed_rows
        df = df[(df["word_count"]>30) & (df["word_count"]<15000)]

        # Repetitiveness
        df['repetitiveness'] = df['text'].apply(lambda t: repetitiveness_score(t, n=3))
        dropped_excessive_repetition = df[~(df['repetitiveness'] < 0.7)]
        self.removed_rows = pd.concat([self.removed_rows, dropped_excessive_repetition])
        df = df[df['repetitiveness'] < 0.7]

        # Remove text which doesn't have enough stopwords to be cohesive
        stopwords = {"the", "be", "to", "of", "and", "that", "have", "with"}
        # Flag rows with no stopwords
        df['no_stopwords'] = df['text'].apply(
            lambda x: not any(word.lower() in stopwords for word in x.split()))
        # filter on this
        dropped_no_stopwords = df[~(df['no_stopwords'] == False)]
        self.removed_rows = pd.concat([self.removed_rows, dropped_no_stopwords])
        df = df[df['no_stopwords'] == False]

        # keep only two cols
        df = df[['text', 'url']]
        print(f"Quality step Shape is: {df.shape}")
        return df