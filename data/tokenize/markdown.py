import typing
import spacy

from data.tokenize import TokenizerInterface

class MarkdownTokenizer(TokenizerInterface):
    spacy_eng = spacy.load('en_core_web_lg')

    def __call__(self, text) -> typing.Sequence[str]:
        text = text.replace('#', ' ').strip()
        return [tok.text.lower() for tok in self.spacy_eng.tokenizer(text)]