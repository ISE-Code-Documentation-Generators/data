import abc
import typing



class SummarizerInterface(abc.ABC):

    @abc.abstractmethod
    def __call__(self, text: str) -> str:
        pass


def get_gensim_summarizer() -> 'SummarizerInterface':
    from ise_cdg_data.summarize.gensim import GensimSummarizer
    return GensimSummarizer()


def get_sumy_summarizer(num_sentences) -> 'SummarizerInterface':
    from ise_cdg_data.summarize.sumy import SumySummarizer
    return SumySummarizer(num_sentences)
