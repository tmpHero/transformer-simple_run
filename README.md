基于transformer 的简易翻译模型

简易封装了一下(目前仅单卡)

抄的项目

https://github.com/hinesboy/transformer-simple

分词器使用 nltk 和 jieba，如果要更改

在 ./YaDeYaDe/text_tokenize_tool.py 里重构 source_text_tokenize 和 result_text_tokenize 函数

自带了英语分词，中文分词，简易分词

更多参数在./YaDeYaDe/parser.py 里修改

## environment

`pip install torch --index-url https://download.pytorch.org/whl/cu121`

`pip install -r requirements.txt`

## Train

```python
from YaDeYaDe.run_utils import *

Train(
	train_file_path="data/train.txt", 
	dev_file_path="data/dev.txt",
	save_path='output/model.pt',
	epochs=5
).start()
```

## Inf

随便注释也复制过来了

```python
from YaDeYaDe.run_utils import *

"""
    Load the model and use

    Attributes:
        model_path (str): model 路径

    Methods:
        translate_text: 翻译文本
            @param input_text: Input text
            @param timeout: Timeout period
	    @param mode: Returns the Boolean type or an error directly
            @return: str or False
"""


translate = Translate('output/model.pt')
res = translate.translate_text('take care.')
print(res)
```
