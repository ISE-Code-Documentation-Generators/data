import markdown
import re

from ise_cdg_data.dataset.interface import Md4DefDatasetInterface

def extract_headers(markdown_text):
    # Parse the Markdown text using the markdown package
    html_text = markdown.markdown(markdown_text)
    
    # Use regular expressions to extract headers from the HTML
    header_tags = re.findall(r'<h(\d)>(.*?)<\/h\1>', html_text)
    
    headers = []
    for tag, text in header_tags:
        level = int(tag)
        headers.append({'level': level, 'text': text})
    
    return headers

def test_extract_headers():
    # Example Markdown text
    markdown_text = """
    # This is a Level 1 Header
    pjoijioji
    ## This is a Level 2 Header


    plkm

    kmm
    ### This is a Level 3 Header
    #### This is a Level 4 Header
    """

    headers = extract_headers(markdown_text)

    for header in headers:
        print(f"Level {header['level']}: {header['text']}")




import pandas as pd

class CNN2RNNDatasetWithPreprocess(Md4DefDatasetInterface):
    source_column = 'source'
    markdown_column = 'markdown'

    def __init__(self, path: str):
        super().__init__()
        self.path = path

        df = pd.read_csv(self.path)
        df = df[[self.source_column, self.markdown_column]]
        df = self.add_header_column(df)
        self.df = df
        print(self.df.head())

    def add_header_column(self, df):
        markdown_headers = df[self.markdown_column].apply(extract_headers)
        markdown_headers = markdown_headers.apply(lambda headers: list(map(lambda header: header['text'], headers)))
        df = df[markdown_headers.apply(lambda headers: len(headers) != 0)]
        df = df.assign(header=markdown_headers).explode('header')
        return df
       
    
    def filter_source(self, tokenizer, max_length):
      tokenized_rows = self.sources.apply(tokenizer.tokenize).apply(len)
      self.df = self.df[tokenized_rows <= max_length]

    def filter_markdown(self, tokenizer):
      tokenized_rows = self.markdowns.apply(tokenizer.tokenize).apply(len)
      tokenized_rows = tokenized_rows.sort_values()
      max_length = tokenized_rows.iloc[math.floor(len(self) *  0.95)]
      self.df = self.df[tokenized_rows <= max_length]
