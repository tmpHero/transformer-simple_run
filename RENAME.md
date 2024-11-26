简易封装了一下(目前仅单卡)

抄的项目

https://github.com/hinesboy/transformer-simple


分词器使用 nltk，如果要更改在 prepare_data.py 里 PrepareData.load_data 修改

jieba等其他库有待试验


更多参数在./YaDeYaDe/parser.py 里修改

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
            @return: str or False
"""


translate = Translate('output/model.pt')
res = translate.translate_text('take care.')
print(res)
```
