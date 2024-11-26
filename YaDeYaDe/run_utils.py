import copy
import os
import gc
import pickle

from .model.attention import MultiHeadedAttention
from .model.position_wise_feedforward import PositionwiseFeedForward
from .model.embedding import PositionalEncoding, Embeddings
from .model.transformer import Transformer
from .model.encoder import Encoder, EncoderLayer
from .model.decoder import Decoder, DecoderLayer
from .model.generator import Generator

import torch
import torch.nn as nn
import torch.nn.functional as F

# global args
from .parser import *
from .prepare_data import PrepareData
from .evaluate import translate_sentence
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from .train import *

from .lib.criterion import LabelSmoothing
from .lib.optimizer import NoamOpt


def print_f(*args, **kwargs):
    print(*args, **kwargs)


def init_model(src_vocab, tgt_vocab, N = 6, d_model = 512, d_ff = 2048, h = 8, dropout = 0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model).to(args.device)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(args.device)
    position = PositionalEncoding(d_model, dropout).to(args.device)
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(args.device), N).to(args.device),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout).to(args.device), N).to(args.device),
        nn.Sequential(Embeddings(d_model, src_vocab).to(args.device), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(args.device), c(position)),
        Generator(d_model, tgt_vocab)).to(args.device)
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model.to(args.device)



"""
    Train
    Methods:
        start: start train
        

    update:
        join: stop train
"""
class Train:
    def __init__(self, train_file_path: str, dev_file_path: str, save_path: str, epochs: int):
        args.train_file, args.dev_file, args.save_file = \
            train_file_path, dev_file_path, save_path, epochs
        self.__data = PrepareData()
        self._save_data_as_pickle()
        self.__model = self._load_model()

    
    def start(self):
        print_f(">>> train start\n")
        criterion = LabelSmoothing(args.tgt_vocab, padding_idx = 0, smoothing= 0.0)
        optimizer = NoamOpt(args.d_model, 1, 2000, torch.optim.Adam(self.__model.parameters(), lr=0, betas=(0.9,0.98), eps=1e-9))
        train(self.__data, self.__model, criterion, optimizer)
        print_f(">>> train end\n")


    def join(self):
        ...

    
    def _save_data_as_pickle(self) -> None:
        export_data = {
            "en_word_dict": self.__data.en_word_dict,
            "cn_word_dict": self.__data.cn_word_dict,
            "cn_index_dict": self.__data.cn_index_dict
        }
        save_folder = os.path.dirname(args.save_file)
        os.makedirs(save_folder, exist_ok=True)
        save_file_path = os.path.join(save_folder, 'word_dict.pkl')
        with open(save_file_path, 'wb') as f:
            pickle.dump(export_data, f)

    def _load_model(self):
        args.src_vocab = len(self.__data.en_word_dict)
        args.tgt_vocab = len(self.__data.cn_word_dict)
        model = init_model(
                            args.src_vocab, 
                            args.tgt_vocab, 
                            args.layers, 
                            args.d_model, 
                            args.d_ff,
                            args.h_num,
                            args.dropout
                        )
        return model


"""
    设置函数超时响应的类装饰器
    Attributes:
        timeout (flaot): 等待时间
        mode (bool): 如果为 False 超时的时候返回 False， 如果为 True 超时的时候 TimeoutError
"""
class TimeoutDecorator:
    def __init__(self, timeout: float, mode: bool = False):
        self.timeout = timeout
        self.mode = mode

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=self.timeout)
                except TimeoutError:
                    if self.mode:
                        raise TimeoutError(f"Function {func.__name__} exceeded time limit of {self.timeout} seconds")
                    return False
        return wrapper



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
class Translate:
    def __init__(self, model_path: str="./model.pt"):
        word_file_path: str = os.path.join(os.path.dirname(model_path), 'word_dict.pkl')
        if not os.path.exists(model_path) and not os.path.exists(word_file_path):
            raise FileNotFoundError(f"{model_path} OR {word_file_path}???")
        
        # self.__data = PrepareData()
        self.__data = self._load_word(word_file_path)
        self.__model = self._load_model(model_path)

    def translate_text(self, input_text: str, timeout:float = 5.0) -> str|bool:
        text_len: int = len(input_text)
        # 多少字符多少时间需要测试, 误差时间默认 1 s
        # words_per_second: float = 12
        # time_error_value: float = 1
        # timeout: float = text_len / words_per_second + time_error_value

        @TimeoutDecorator(timeout=timeout)
        def _translate_text():
            return translate_sentence(self.__model, input_text, self.__data, args)
        return _translate_text()

    def _load_word(self, word_file_path):
        with open(word_file_path, 'rb') as f:
            return pickle.load(f)

    def _load_model(self, model_path):
        args.src_vocab = len(self.__data['en_word_dict'])
        args.tgt_vocab = len(self.__data['cn_word_dict'])
        model = init_model(
                            args.src_vocab, 
                            args.tgt_vocab, 
                            args.layers, 
                            args.d_model, 
                            args.d_ff,
                            args.h_num,
                            args.dropout
                        )
        model.load_state_dict(torch.load(model_path, weights_only=False))
        model.eval()
        return model
    

    


    