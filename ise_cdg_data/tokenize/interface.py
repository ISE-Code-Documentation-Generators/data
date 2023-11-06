import abc
import typing

class TokenizerInterface(abc.ABC):

    @abc.abstractmethod
    def __call__(self, text) -> typing.Sequence[str]:
        pass


def get_source_and_markdown_tokenizers(cleanse_markdown=True) -> typing.Tuple[TokenizerInterface, TokenizerInterface]:
    from ise_cdg_data.tokenize.text import MarkdownToTextTokenizer, TextTokenizer
    from ise_cdg_data.tokenize.source import SourceTokenizer
    if cleanse_markdown:
        markdown_tokenizer = MarkdownToTextTokenizer()
    else:
        markdown_tokenizer = TextTokenizer()

    return SourceTokenizer(), markdown_tokenizer