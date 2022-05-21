# VineFinder
VineFinder is a UNet model that can be used to find the optimal picking point for vine tomatoes for automating the picking.

The images are of various width and height and contain, in general, a single vine tomatoe: ![image of vine tomatoe](docs/image_0_0.png)

## Results

## Configuring DVC
[Install DVC](https://dvc.org/doc/install) and then run the following command:
```
dvc pull
```

## Usage
To train the model, you need to download the dataset first (see section above).

Run the model with the following command:
```
    python3 test.py -c configTest.json -r saved/.../model.pth
```
