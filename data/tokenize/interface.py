import abc
import typing

class TokenizerInterface(abc.ABC):

    @abc.abstractmethod
    def __call__(self, text) -> typing.Sequence[str]:
        pass


def get_source_and_markdown_tokenizers() -> typing.Tuple[TokenizerInterface, TokenizerInterface]:
    from data.tokenize.markdown import MarkdownTokenizer
    from data.tokenize.source import SourceTokenizer

    return SourceTokenizer(), MarkdownTokenizer()