import typing
import spacy

from ise_cdg_data.tokenize import TokenizerInterface
from ise_cdg_data.tokenize.md2text import MarkdownToText

class TextTokenizer(TokenizerInterface):
    spacy_eng = spacy.load('en_core_web_lg')

    def __call__(self, text) -> typing.Sequence[str]:
        return [tok.text.lower() for tok in self.spacy_eng.tokenizer(text) if not tok.is_space]

class MarkdownToTextTokenizer(TokenizerInterface):
    spacy_eng = spacy.load('en_core_web_lg')
    md2text = MarkdownToText()
    
    def __call__(self, text) -> typing.Sequence[str]:
        return [tok.text.lower() for tok in self.spacy_eng.tokenizer(self.md2text(text))]
