import pandas as pd
from pipeline import PipelineStep
import hashlib
from datasketch import MinHash, MinHashLSH

def hash_text(text):
    """
    Apply exact hashing to df
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def assign_shard(fp, n_shards=8):
    """Assign text to a shard based on hashing"""
    return int(fp, 16) % n_shards

def shard_dataframe(df, n_shards=8):
    # Normalise and fingerprint
    df['shard'] = df['hashing'].apply(lambda x: assign_shard(x, n_shards))
    return df

def create_minhash(text):
        # TO DO REWRITE TO TAKE NUM PERM IN FUNCTION
        m = MinHash(num_perm=128)
        for word in text.lower().split():
            m.update(word.encode('utf-8'))
        return m

def split_paragraphs(df):
    # create a docindex for split
    df['doc_id'] = df.index
    all_paragraphs = []

    for idx, row in df.iterrows():
        doc_id = row['doc_id']
        text = row['text']
        url = row['url']


        # split by para
        paragraphs = text.split("\n\n")

        # non empty paras
        paragraphs = [para.strip() for para in paragraphs if para.strip()]
        # useing a dict store para info in all_paragraphs
        # paragraph id important for order within docs
        for i, para in enumerate(paragraphs):
            all_paragraphs.append({
                'doc_id': doc_id,
                'paragraph_id': i,
                'paragraph_text': para,
                'url': url
                })
    df_paragraphs = pd.DataFrame(all_paragraphs)
    return df_paragraphs


class ExactDeDuplicationStep(PipelineStep):
    def __init__(self, name, validator):
        super().__init__(name, validator)
    
    def run(self, df: pd.DataFrame):
        """
        Use hashing algorithm on df text for exact matches
        """
        df['hashing'] = df['text'].apply(hash_text)

        n_shards = 8
        df = shard_dataframe(df, n_shards=n_shards)
        deduped_list = []
        dropped_list = []

        for shard_id in range(n_shards):
            # group text by shard
            shard_df = df[df['shard'] == shard_id].copy()
            before = len(shard_df) # shard length beforehand

            #Marking duplicates
            mask_duplicates = shard_df.duplicated(subset=['hashing'], keep='first')
            
            # inversing to get original (first duplicate) (inverse of boolean)
            shard_dedup = shard_df[~mask_duplicates]
            # Rows that were dropped
            shard_dropped = shard_df[mask_duplicates]

            deduped_list.append(shard_dedup)
            dropped_list.append(shard_dropped)

            after = len(shard_dedup)
            print(f"Shard {shard_id}: removed {before - after} duplicates out of {before}")

        self.removed_rows = pd.concat(dropped_list, ignore_index=True)
        deduped_df = pd.concat(deduped_list, ignore_index=True)

        return deduped_df

class FuzzyDeduplicationStep(PipelineStep):
    def __init__(self, name, validator):
        super().__init__(name, validator)
    
    def run(self, df: pd.DataFrame):
        """
        Use minhash to find and remove fuzzy duplicates on a paragraph basis.
        This splits into parahs, removes fuzzy duplicates then rolls up back to original docs.
        """
        num_perm = 128
        threshold = 0.8
        n_shards = 8

        df_paragraphs = split_paragraphs(df)
        # need to reset index
        df_paragraphs = df_paragraphs.reset_index(drop=True)

        # todo test for nas
        print(df_paragraphs['paragraph_text'].isna().sum()) 
        # hash based sharding not needed for fuzzy match
        df_paragraphs['shard'] = df_paragraphs.index % n_shards
        deduped_list = []
        dropped_list = []

        for shard_id in range(n_shards):
            shard_df = df_paragraphs[df_paragraphs['shard'] == shard_id].copy()
            print(f"Processing shard {shard_id}, {len(shard_df)} paragraphs")

            shard_df['minhash'] = shard_df['paragraph_text'].apply(create_minhash)

            # Build LSH index
            lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
            for idx, mh in enumerate(shard_df['minhash']):
                lsh.insert(f"p{idx}", mh)

            # Detect duplicates
            seen = set()
            to_remove = set()
            for idx, mh in enumerate(shard_df['minhash']):
                if idx in seen:
                    continue
                result = lsh.query(mh)
                result = [int(r[1:]) for r in result if int(r[1:]) != idx]  # remove first
                # mark all other similar paragraphs as duplicates
                to_remove.update(result)

                # mark first instance as seen
                seen.add(idx)
                seen.update(result)  # also mark the removed ones so we skip them in future

            shard_dropped = shard_df.iloc[list(to_remove)].reset_index(drop=True)
            shard_dedup = shard_df.drop(shard_df.index[list(to_remove)]).reset_index(drop=True)

            deduped_list.append(shard_dedup)
            dropped_list.append(shard_dropped)

            print(f"Shard {shard_id}: removed {len(shard_dropped)} duplicate paragraphs")

        df_deduped = pd.concat(deduped_list, ignore_index=True)
        self.removed_rows = pd.concat(dropped_list, ignore_index=True)

        # Wrap the paragraphs back up on doc_id and url
        df_docs_cleaned = (
            df_deduped.groupby('doc_id').agg(
                text=('paragraph_text', lambda x: " ".join(x.sort_values().tolist())),
                url=('url', 'first')  # url also needs to be rolled up
            ).reset_index()
        )

        # todo: validate urls and docid's wraooed correctly
        return df_docs_cleaned



