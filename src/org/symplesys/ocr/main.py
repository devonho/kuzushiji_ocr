from org.symplesys.ocr.train import train
from org.symplesys.ocr.model import model
from org.symplesys.ocr.utils import parseArgs
from org.symplesys.ocr.infer import infer

def main():
    args = parseArgs()
    if args.train:
        device = "cuda"
        train(model, device, "datasets/kuzushiji/", 100, num_epochs=5)
    else:
        if args.image:
            output_label = infer(args.image)
            print(output_label)


if __name__ == "__main__":
    main()