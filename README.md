# Kuzushiji OCR 「くずし字のOCR」

This is a demo that trains a PyTorch model to classify cursive Hiragana. 


# Dataset and Model

The dataset is a MNIST-style [Kuzushiji-MNIST](https://www.kaggle.com/datasets/anokas/kuzushiji/data) that maps Hiragana characters 「お、ら、す、つ、な、は、ま、や、れ、を」 to the 10 classes. There are 60K and 10K examples for the `train` and `test` phases respectively. 

The model used is a [Resnet](https://pytorch.org/vision/stable/models/resnet.html) CNN that has a small memory foot-print, and can fit into legacy consumer GPUs like a 11Gb GTX1080Ti.


# Running 

Create and activate virtualenv

```
virtualenv myenv
source ./myenv/bin/activate
pip install -r requirements.txt
```
Download dataset

```
mkdir ./datasets
curl -L -o ./datasets/kuzushiji.zip \  
    https://www.kaggle.com/api/v1/datasets/download/anokas/kuzushiji
unzip ./datasets/kuzushiji.zip -d ./datasets/kuzushiji

```

Train model

```
PYTHONPATH=./src python -m org.symplesys.ocr.main --train 
```

Test model

```
PYTHONPATH=./src python -m org.symplesys.ocr.main --image "./datasets/kuzushiji_images/test/00003.png"
```

# Notebooks

* [eda.ipynb](./notebooks/eda.ipynb) - Exploration of the Kuzushiji MNIST dataset
* [infer.ipynb](./notebooks/infer.ipynb) - Inference using trained model
* [etl.ipynb](./notebooks/etl.ipynb) - ETL for GCP Vertex AI