# Magnetic Tile Defect Classifier

A Streamlit app that classifies magnetic tile images as defective or not, using a fine-tuned DenseNet121 (or VGG16, or ResNet18). Built for INFO 4160 Industrial Internet of Things. Image in → CNN → defect label out.

## What it does

The user uploads a tile image. The app runs it through a transfer-learned CNN trained on the magTileDefectData dataset (6 classes, ~1,300 images). The model predicts a class and confidence; if confidence > 0.5 the tile is flagged as defective. The dropdown lets you swap between three architectures — ResNet18, VGG16, and DenseNet121 — to compare model performance on the same input.

Training freezes everything except the final classifier layer and fine-tunes for 10 epochs.

## Stack

- `torch` and `torchvision` for the CNNs
- `streamlit` for the upload + classify UI
- Transfer learning from ImageNet weights, final layer fine-tuned
- `MachineVisionModel` helper class wraps the train/eval loop

## Running it

Train at least one model (the app expects `.pth` files in the project root):

```bash
pip install -r requirements.txt
jupyter notebook mp1_stuff.ipynb   # produces vgg_full.pt
```

Then launch the app:

```bash
streamlit run MP2.py
```

You will need the `magTileDefectData` dataset in the project root with the standard `train/val/test` folder structure.

## What I struggled with

The biggest issue I had was training the models on my older laptop. As it was around 6 to 7 years old at the time, it was having trouble training the models at a relatively quick rate and could easily crash, resetting my progress completely.

## Files

```
MP2.py                            Streamlit app, model dropdown + classification
mp1_stuff.ipynb                   training notebook 
helper_CNN/machine_vision_model1.py   the MachineVisionModel class
```
