import re

import markdown as mdutils # pip install markdown
from bs4 import BeautifulSoup # pip install beautifulsoup4


class MarkdownToText:

    def md_to_text(cls, md: str) -> str:
        html = mdutils.markdown(md)
        soup = BeautifulSoup(html, features='html.parser')
        return soup.get_text()
    
    def remove_codes_and_formula(cls, markdown_text: str) -> str:
        # Remove code blocks (e.g., `code`)
        markdown_text = re.sub(r'```([^`]+)```', r'', markdown_text)

        # Remove code blocks (e.g., `code`)
        markdown_text = re.sub(r'`([^`]+)`', r'', markdown_text)

        # Remove images (e.g., ![alt text](image url))
        markdown_text = re.sub(r'!\[([^\]]+)\]\([^)]+\)', r'', markdown_text)

        # Remove block formulas ($$...$$)
        markdown_text = re.sub(r'\$\$.*(?<!\n)\$\$', '', markdown_text)
        
        # Remove inline formulas ($...$)
        markdown_text = re.sub(r'\$.*(?<!\n)\$', '', markdown_text)

        return markdown_text

    def __call__(self, markdown: str) -> str:
        return self.remove_codes_and_formula(self.md_to_text(markdown))
