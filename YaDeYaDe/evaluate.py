import torch
import numpy as np
import time

from .parser import args
from torch.autograd import Variable
from .utils import subsequent_mask

from nltk import word_tokenize

def log(data, timestamp):
    file = open(f'log/log-{timestamp}.txt', 'a')
    file.write(data)
    file.write('\n')
    file.close()

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


def evaluate(data, model):
    timestamp = time.time()
    with torch.no_grad():
        for i in range(len(data.dev_en)):
            en_sent = " ".join([data.en_index_dict[w] for w in data.dev_en[i]])
            print("\n" + en_sent)
            log(en_sent, timestamp)
            cn_sent = " ".join([data.cn_index_dict[w] for w in data.dev_cn[i]])
            print("".join(cn_sent))
            log(cn_sent, timestamp)

            src = torch.from_numpy(np.array(data.dev_en[i])).long().to(args.device)
            src = src.unsqueeze(0)
            src_mask = (src != 0).unsqueeze(-2)

            out = greedy_decode(model, src, src_mask, max_len = args.max_length, start_symbol = data.cn_word_dict["BOS"])

            translation = []
            for j in range(1, out.size(1)):
                sym = data.cn_index_dict[out[0, j].item()]
                if sym != 'EOS':
                    translation.append(sym)
                else:
                    break
            print("translation: %s" % " ".join(translation))
            log("translation: " + " ".join(translation) + "\n", timestamp)




def translate_sentence(model, en_sentence, data, args):
    # 1. 英文句子的分词和ID化
    en_tokens = ["BOS"] + word_tokenize(en_sentence.lower()) + ["EOS"]
    en_ids = [data['en_word_dict'].get(w, data['en_word_dict']['UNK']) for w in en_tokens]
    
    # 2. 创建源语言输入张量并构建掩码
    src = torch.tensor(en_ids).unsqueeze(0).long().to(args.device)
    src_mask = (src != 0).unsqueeze(-2)  # 生成源语言的掩码，避免填充部分
    
    # 3. 使用贪心解码生成翻译
    with torch.no_grad():
        out = greedy_decode(model, src, src_mask, args.max_length, start_symbol=data['cn_word_dict']["BOS"])

    # 4. 将生成的目标语言索引转换为中文
    translation = []
    for j in range(1, out.size(1)):  # 跳过BOS标记
        word = data['cn_index_dict'][out[0, j].item()]
        if word != 'EOS':
            translation.append(word)
        else:
            break
    
    # 5. 返回翻译的中文句子
    return "".join(translation)
