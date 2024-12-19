from nltk import word_tokenize
import jieba

def source_text_tokenize(input_text: str) -> list[str]:
    return word_tokenize(input_text.lower())



def result_text_tokenize(input_text: str) -> list[str]:
    # return word_tokenize( " ".join([w for w in input_text]) )
    return [word for word in jieba.cut(input_text)]
