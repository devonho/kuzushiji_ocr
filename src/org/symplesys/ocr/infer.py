from PIL import Image
import numpy as np
import torch

from org.symplesys.ocr.model import model

def load_model(checkpoint):
    params = torch.load(checkpoint, weights_only=True)
    model.load_state_dict(params)
    return model

def infer(filename, checkpoint="./checkpoints/best_model_params.pt"):
    with Image.open(filename) as img:
        arr = np.array(img)

    image = np.zeros([3*28*28]).reshape([3,28,28])
    image[0,:,:] = arr
    image[1,:,:] = arr
    image[2,:,:] = arr
    ts = torch.Tensor(image.astype(np.double)).reshape([1,3,28,28])

    trained_model = load_model(checkpoint)
    trained_model.eval()
    with torch.no_grad():
        res = trained_model(ts).relu()
    index = torch.where(res == res.max())[1]
    output_label = int(index)
    return output_label