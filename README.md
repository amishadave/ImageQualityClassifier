# Image Quality Classifier

This classifier identifies whether an adaptive optics image is of good (0) or poor (1) quality.

## Getting Started

You can run the project using `python3 script.py`

The dataset is specified at the end of `script.py`. The dataset can be easily swapped to a new dataset by changing the paths and associated parameters.

## Create a Model Based on Annotated Data

Ensure lines 173 and 178 are commented out and lines 180 and 181 are uncommented. Images should be in a folder called "datasets" with an excel sheet that has a list of the file names with their annotated classification (good=0, poor=1). The script should be in the same folder as "datasets".

## Load a saved model with New Data
Ensure lines 180 and 181 are commented out and lines 173 and 178 are uncommented. New images to be analyzed should be in a folder called "new" within the "datasets" folder. The "new" folder should have an excel sheet with a list of file names. The script should be in the same folder as "datasets".

## Dependencies
This project is dependent on scikit-learn, scikit-image, matplotlib, seaborn, pandas, numpy, and opencv.

## Resources Used
https://rpubs.com/Sharon_1684/454441

https://scikit-learn.org/0.15/modules/generated/sklearn.svm.libsvm.predict_proba.html

https://stackoverflow.com/questions/15564410/scikit-learn-svm-how-to-save-load-support-vectors

Tutorial Code from my Applied Machine Learning FAES class