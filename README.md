# Image Quality Classifier

This classifier identifies whether an adaptive optics image is of good (0) or poor (1) quality.

## Getting Started

You can run the project using `python3 script.py`

The dataset is specified at the end of `script.py`. The dataset can be easily swapped to a new dataset by changing the paths and associated parameters.

The model is also specified at the end of `script.py`. The model can also easily be swapped to a different model. You can use the code with your own data to generate your own model by uncommenting the last 2 lines of the code (lines 165, 166) and commenting out lines 158 and 163. A  model you may already have or one you just generated using the code can be loaded at the end of the script (line 158).

## Dependencies
This project is dependent on scikit-learn, scikit-image, matplotlib, seaborn, pandas, numpy, and opencv.

## Resources Used
https://rpubs.com/Sharon_1684/454441

https://scikit-learn.org/0.15/modules/generated/sklearn.svm.libsvm.predict_proba.html

https://stackoverflow.com/questions/15564410/scikit-learn-svm-how-to-save-load-support-vectors

https://elitedatascience.com/imbalanced-classes

https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76

https://github.com/WillKoehrsen/Machine-Learning-Projects/blob/master/Random%20Forest%20Tutorial.ipynb

Tutorial Code from my Applied Machine Learning FAES class