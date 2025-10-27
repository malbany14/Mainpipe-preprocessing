## Setup Data

This project is originally based on the `mainpipe_data_v1.jsonl` dataset.

Download the dataset from: 
   [https://s3.us-east-1.amazonaws.com/mainpipe.maincode.com/mainpipe_data_v1.jsonl](https://s3.us-east-1.amazonaws.com/mainpipe.maincode.com/mainpipe_data_v1.jsonl)

Place it in the `data/raw/` folder:

Alternatively place any raw data ready for pre-processing in the data/raw folder. The data structure this works on is ['text']['url']


### Install Dependencies

Use a virtual environment (windows bash):

python -m venv venv

venv\Scripts\activate

pip install --upgrade pip

pip install -r requirements.txt

## Running the Pipeline

Once the dataset is in `data/raw/`, you can run the full preprocessing pipeline. The main orchestration script is located in `mainpipe/Pipeline/main.py`.

## Output

The tokenised dataset will be generated under 'data/cleaned' as 'cleaned_csv_test_tokens.npy'

The cleaned text dataset will also be generated in 'data/cleaned' as 'cleaned_csv_test.JSONL'

The overall pipeline report as well as csv's of dropped rows will be generated in 'reports'