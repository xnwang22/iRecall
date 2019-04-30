import iRecall

if __name__ == "__main__":
    dictionary = iRecall.load_dictionary('./data/')
    maxConfidence = 10000
    iRecall.process_image('test.jpg', dictionary, './data/', maxConfidence)
