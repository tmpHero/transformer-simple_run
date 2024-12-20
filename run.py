from YaDeYaDe.run_utils import *




if __name__ == "__main__":

    Train(
        train_file_path="data/ja/train.txt", 
        dev_file_path="data/ja/dev.txt",
        save_path='output/zh_ja/model.pt',
        epochs=5
    ).start()


    translate = Translate('output/zh_ja/model.pt')
    for text in ["你好。", "我是谁？", "吃饭了吗？"]:
        res = translate.translate_text(text)
        print(res)
