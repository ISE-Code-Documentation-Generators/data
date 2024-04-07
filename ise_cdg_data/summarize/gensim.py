from ise_cdg_data.summarize.interface import SummarizerInterface


class GensimSummarizer(SummarizerInterface):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, text: str) -> str:
        try:
            from gensim.summarization import summarize

            return summarize(text)
        except ImportError:
            raise Exception("gensim is not installed!")
