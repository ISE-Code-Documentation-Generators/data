import typing
from pygments import lex, token as pytoken
from pygments.lexers import PythonLexer
from pygments.token import Token

from data.tokenize import TokenizerInterface


class SourceTokenizer(TokenizerInterface):
    ignores = [
        pytoken.Text,
        pytoken.Comment,
        pytoken.Punctuation,
        pytoken.Escape,
        pytoken.Error,
        pytoken.Other,
        Token.Comment,
        Token.Literal,
    ]
    def should_ignore_token(self, t: typing.Tuple[str, pytoken._TokenType]):
        for ignore in self.ignores:
            if str(ignore) in str(t[1]):
                return True
        return False

    def __call__(self, text) -> typing.Sequence[str]:
        tokens = lex(text, PythonLexer())
        tokens = [(token[1], token[0]) for token in tokens]
        final_tokens = []

        for t in tokens:
            if not self.should_ignore_token(t):
                final_tokens.append(t[0])

        return final_tokens
