import typing
from enum import Enum

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers import AbstractSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer

from ise_cdg_data.summarize.interface import SummarizerInterface


class SumySummarizer(SummarizerInterface):

    class Algorithm(Enum):
        lsa = LsaSummarizer
        text_rank = TextRankSummarizer
        lex_rank = LexRankSummarizer
        luhn = LuhnSummarizer

        default = text_rank

    def __init__(
        self, num_sentences: int = 1, algorithm: typing.Optional["Algorithm"] = None
    ) -> None:
        super().__init__()
        self.num_sentences = num_sentences
        algorithm = algorithm or self.Algorithm.default
        self.summarizer: AbstractSummarizer = algorithm.value()

    def __call__(self, text: str) -> str:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summary_sentences = self.summarizer(parser.document, self.num_sentences)
        return " ".join(str(x) for x in summary_sentences)
