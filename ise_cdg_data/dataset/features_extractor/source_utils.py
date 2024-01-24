
# extract number of python operators from text
def extract_operator_count(text):
    operators = [
        "+",
        "-",
        "*",
        "/",
        "**",
        "%",
        "//",
        "=",
        "+=",
        "-=",
        "*=",
        "/=",
        "%=",
        "//=",
        "**=",
        "&=",
        "|=",
        "^=",
        ">>=",
        "<<=",
        "==",
        "!=",
        ">=",
        "<=",
        ">",
        "<",
        " and ",
        " or ",
        " not ",
        " is ",
        " is not ",
        " in ",
        " not in ",
        "&",
        "|",
        "^",
        "~",
        "<<",
        ">>",
    ]
    count = 0
    for operator in operators:
        count += text.count(operator)
    return count


def extract_unique_operator_count(text):
    operators = [
        "+",
        "-",
        "*",
        "/",
        "**",
        "%",
        "//",
        "=",
        "+=",
        "-=",
        "*=",
        "/=",
        "%=",
        "//=",
        "**=",
        "&=",
        "|=",
        "^=",
        ">>=",
        "<<=",
        "==",
        "!=",
        ">=",
        "<=",
        ">",
        "<",
        " and ",
        " or ",
        " not ",
        " is ",
        " is not ",
        " in ",
        " not in ",
        "&",
        "|",
        "^",
        "~",
        "<<",
        ">>",
    ]
    unique_count = 0
    for operator in operators:
        if text.count(operator) > 0:
            unique_count += 1
    return unique_count


# remove file address from string
import re

import pandas as pd
def remove_file_address(text):
    text = re.sub(r"\S*[\\|/]\S*", "", text)
    return text


def remove_ampersand(text):
    text = re.sub(r"%[a-zA-Z]+", "", text)
    return text


def remove_extras(text):
    text = remove_ampersand(text)
    text = remove_file_address(text)
    return text

#extract python keywords from text
def extract_keywords_count(text):
    keywords = ['True', 'False', 'None', 'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']
    count = 0
    for keywords in keywords:
        count += text.count(keywords)
    return count

keywords = ['True', 'False', 'None', 'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']

def extract_identifier_count(text):
    count = 0
    identifiers = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text)
    for identifier in identifiers:
        if identifier not in keywords:
            count += 1
    return count

def extract_avg_len_identifier(text):
    iden_lens = []
    identifiers = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text)
    for identifier in identifiers:
        if identifier not in keywords:
            iden_lens.append(len(identifier))
    if len(iden_lens) == 0:
        return 0
    return sum(iden_lens) // len (iden_lens)

def extract_max_len_identifier(text):
    iden_lens = []
    identifiers = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text)
    for identifier in identifiers:
        if identifier not in keywords:
            iden_lens.append(len(identifier))
    if len(iden_lens) == 0:
        return 0
    return max(iden_lens)


# Klemola, T. & Rilling, Juergen. (2003). A cognitive complexity metric based on category learning.

def extract_unique_lines_of_code(text):
    lines = text.split('\n')
    return set(lines)

def klcid(text):
    unique_lines_code = extract_unique_lines_of_code(text)
    iden_count = []
    for line in unique_lines_code:
        iden_count.append(extract_identifier_count(line))
    unique_lines_code_with_iden = []
    for line in unique_lines_code:
        if extract_identifier_count(line) > 0:
            unique_lines_code_with_iden.append(line)
    if len(unique_lines_code_with_iden) == 0:
        return 0
    return sum(iden_count) / len(unique_lines_code_with_iden)

# Extract operands

def extract_operand_count(string):
    res = re.split(r'''[\-()\+\*\/\=\&\|\^\~\<\>\%]|
    \*\*|\/\/|
    \>\>|\<\<|
    \+\=|\-\=|\*\=|\/\=|\%\=|
    \/\/\=|\*\*\=
    \&\=|\|\=|\^\=|\>\>\=|\>\>\=|
    \=\=|\!\=|\>\=''', string)
    res = list(filter(None, res))
    return len(res)

def extract_unique_operand_count(string):
    res = re.split(r'''[\-()\+\*\/\=\&\|\^\~\<\>\%]|
    \*\*|\/\/|
    \>\>|\<\<|
    \+\=|\-\=|\*\=|\/\=|\%\=|
    \/\/\=|\*\*\=
    \&\=|\|\=|\^\=|\>\>\=|\>\>\=|
    \=\=|\!\=|\>\=''', string)
    res = list(filter(None, res))
    return len(set(res))

# Count python arguments

def python_arguments(string):
    args = []
    args_regex = re.compile(
            r'''(
                [a-zA-Z_][a-zA-Z0-9_]*\((.*?)\)
            )''',
            flags=re.DOTALL | re.VERBOSE | re.MULTILINE
        )
    try:
        s = re.findall(args_regex, string)
        z = [i[0] for i in s]
        for i in z:
            args = args + re.search(r'\((.*?)\)',i).group(1).split(',')
        return len(args)
    except:
        return 0

# Loop and condition statements
    
def extract_loop_statements_count(text):
    keywords = ['while', 'for']
    count = 0
    for keywords in keywords:
        count += text.count(keywords)
    return count

def extract_if_statements_count(text):
    keywords = ['if']
    count = 0
    for keywords in keywords:
        count += text.count(keywords)
    return count

def statements_count(text):
    return extract_loop_statements_count(text) + extract_if_statements_count(text)

# nested depth block
def nested_depth(text):
    blocks = text.split('\n')
    depth = []
    for block in blocks:
        depth.append(block.count(' '))
    return sum(depth) / len(depth)

import radon
from radon.visitors import ComplexityVisitor
def complexity_analysis(source):
    v = ComplexityVisitor.from_code(source)
    return v.complexity
def complexity_analysis2(source):
    source_new = source.replace('%matplotlib inline', '')
    try:
        v = ComplexityVisitor.from_code(source_new)
        return v.complexity
    except Exception as e:
        # print(e)
        return 0
    
### External API Popularity
def capture_imports(source_code):
    import_regex = r"^\s*(?:from|import)\s+(\w+(?:\s*,\s*\w+)*)"
    # Find all import statements
    import_matches = re.findall(import_regex, source_code, re.MULTILINE)
    for i in import_matches:
        if ',' in i:
            new_i = i.replace(' ', '').split(',')
            import_matches.extend(new_i)
            import_matches.remove(i)
    return import_matches

import collections

def eap_score_function_generator(api_column: "pd.Series"):
    l = api_column.values.tolist()
    flat_list = [item for sublist in l for item in sublist]
    print("flat_list")
    print(flat_list)
    eap_dict = dict(collections.Counter(flat_list))
    print("eap_dict")
    print(eap_dict)
    eap_score_dict = dict(sorted(eap_dict.items(), key=lambda item: item[1], reverse=True))
    # TODO should find max_freq based on the data
    print("eap_score_dict")
    print(eap_score_dict)
    max_freq = eap_dict['sklearn']
    for k, v in eap_score_dict.items():
        eap_score_dict[k] = v / max_freq
    def eap_score(ap_list):
        score = 0
        for i in ap_list:
            score += eap_score_dict.get(i, 0)
        return score
    return eap_score

### comment metrics

#### Count inline comments

def count_inline_comment(string: str) -> str:
    inline_regex = re.compile(
        r'''(
            (?<=\#).+ # comments like: # This is a comment
        )''',
        flags=re.VERBOSE
    )
    # return inline_regex.sub('', string)

    return len(re.findall(inline_regex, string))

def multi_line_comments(string: str) -> str:

    # Python comments
    multi_line_python_regex = re.compile(
        r'''(
            (?<=\n)\'{3}.*?\'{3}(?=\s*\n) |
            (?<=^)\'{3}.*?\'{3}(?=\s*\n) |
            (?<=\n)\'{3}.*?\'{3}(?=$) |
            (?<=^)\'{3}.*?\'{3}(?=$) |
            (?<=\n)\"{3}.*?\"{3}(?=\s*\n) |
            (?<=^)\"{3}.*?\"{3}(?=\s*\n) |
            (?<=\n)\"{3}.*?\"{3}(?=$) |
            (?<=^)\"{3}.*?\"{3}(?=$)
        )''',
        flags=re.DOTALL | re.VERBOSE | re.MULTILINE
    )
    python_multi_line_count = re.findall(multi_line_python_regex, string)


    return python_multi_line_count

def extract_line_comments(text):
    multi_lines_lines = []
    multi_lines = multi_line_comments(text)
    for line in multi_lines:
        multi_lines_lines += line.split('\n')
    return len(multi_lines_lines) + count_inline_comment(text)

def count_comment_word(text):
    comment_word_count = 0
    comments = re.findall(r'(?<=\#).+', text)
    for comment in comments:
        comment_word_count += len(comment.split())
    return comment_word_count