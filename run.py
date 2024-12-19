from YaDeYaDe.run_utils import *




if __name__ == "__main__":

    # Train(
    #     train_file_path="data/train.txt", 
    #     dev_file_path="data/dev.txt",
    #     save_path='output/model.pt',
    #     epochs=8
    # ).start()


    translate = Translate('output/model.pt')
    for text in ["take care.", "hi!", "who are your?"]:
        res = translate.translate_text(text)
        print(res)
