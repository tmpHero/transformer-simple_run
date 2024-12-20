from nltk import word_tokenize
import jieba

def zh_text_tokenize(input_text: str) -> list[str]:
    return [word for word in jieba.cut(input_text)]


def en_text_tokenize(input_text: str) -> list[str]:
    return word_tokenize(input_text.lower())


def simple_text_tokenize(input_text: str) -> list[str]:
    return word_tokenize( " ".join([w for w in input_text]) )



"""
    预处理调用的两个函数
"""
def source_text_tokenize(input_text: str) -> list[str]:
    return zh_text_tokenize(input_text)


def result_text_tokenize(input_text: str) -> list[str]:
    return simple_text_tokenize(input_text)





