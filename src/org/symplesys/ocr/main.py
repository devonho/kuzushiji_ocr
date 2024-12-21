import argparse
from org.symplesys.ocr.train import train
from org.symplesys.ocr.model import model

def training():
    device = "cuda"
    model_trained =  train(model, device, "datasets/kuzushiji/", 100, num_epochs=25)

def inference():
    pass

def main():
    training()

if __name__ == "__main__":
    main()