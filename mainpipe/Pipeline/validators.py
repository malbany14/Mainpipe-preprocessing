import pandas as pd
import re

def general_validations(df:pd.DataFrame):
    """
    Some general df validations that will run at each step
    """
    stats = {}
    nullsintxt = int(df['text'].isna().sum())
    stats['Nulls in text data'] = nullsintxt

    # html checking
    df["element_count"] = df["text"].apply(count_html_tags)
    stats['Html tags'] = int(df['element_count'].sum())

    # utf8 encoding checking
    df['non-utf8_count'] = df["text"].apply(count_non_utf8_chars)
    stats['Utf8 chars'] = int(df['non-utf8_count'].sum())

    return stats

def count_html_tags(text):
    """
    simple regex to get a general sense of the amt of html tags in text
    """
    TAG_REGEX = re.compile(r"<\s*/?\s*([a-zA-Z0-9]+)[^>]*>") # general to match html tags
    return len(TAG_REGEX.findall(text))

def count_non_utf8_chars(text):
    """
    Return count of characters which cant be encoded to utf8
    """
    count = 0
    for c in text:
        try:
            c.encode('utf-8')
        except UnicodeEncodeError:
            count += 1
    return count

class Validator:
    """
    Class to validate df's once they go through a pipeline step
    """
    def __init__(self):
        self.stats = {}

    def validate(self, df: pd.DataFrame):
        """
        Validate the df procudced in the Pipelinestep. Overwritten for other validation types
        """
        raise NotImplementedError

class GeneralValidator(Validator):
    """
    The purpose of this class is to run a bunch of general validation steps
    """
    def __init__(self):
        super().__init__()
    
    def validate(self, df: pd.DataFrame):
        """
        Run general validations
        """
        self.stats = general_validations(df) # always start self.stats with general stats
