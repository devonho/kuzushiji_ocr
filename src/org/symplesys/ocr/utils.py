import os
import pathlib
import argparse

def createFolder(path):
    p = pathlib.Path(path)
    if not p.exists():
        os.mkdir(path)


def parseArgs():
    parser = argparse.ArgumentParser(
                    prog='Kuzushiji OCR',
                    description='OCR Demo',
                    epilog='http://github.com/devonho/kuzushiji_ocr')
    parser.add_argument('-t','--train', action='store_true', required=False) 
    parser.add_argument('-i','--image', required=False) 
    args = parser.parse_args()
    return args