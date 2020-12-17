

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

import cv2
import imutils
# The digits dataset
digits = datasets.load_digits()

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
"""
_, axes = plt.subplots(2, 4)
images_and_labels = list(zip(digits.images, digits.target))
for ax, (image, label) in zip(axes[0, :], images_and_labels[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)
"""
# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# Split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)
#image = [X_test[0][int(8*i):x] for i,x in enumerate(range(0,len(X_test[0]),8))]

#
# We learn the digits on the first half of the digits
classifier.fit(X_train, y_train)

# Now predict the value of the digit on the second half:
predicted = classifier.predict(X_test)

"""
images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Prediction: %i' % prediction)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, predicted)))
disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)
"""


video_capture = cv2.VideoCapture(0)
prediction = ""
thresh = None
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (41, 41), 0)
    if thresh is None:
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                       cv2.THRESH_BINARY, 51, 2)
        width = len(thresh)
        height = len(thresh[0])
        sq = min(width,height)
        crop = thresh[width // 2 - sq // 2:width // 2 + sq // 2, height // 2 - sq // 2:height // 2 + sq // 2]
        crop = thresh[width // 2 - sq // 2:width // 2 + sq // 2, height // 2 - sq // 2:height // 2 + sq // 2]
        crop = cv2.GaussianBlur(crop, (41, 21), 0)
        img = imutils.resize(crop, width=8, height=8)


    k = cv2.waitKey(1)
    if k % 256 == 27:  # ESC Pressed
        break

    if k % 256 == 32:  # SPACE Pressed
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                       cv2.THRESH_BINARY, 51, 2)
        width = len(thresh)
        height = len(thresh[0])
        sq = min(width,height)

        crop = thresh[width // 2 - sq // 2:width // 2 + sq // 2,height // 2 - sq // 2:height // 2 + sq // 2]
        crop = cv2.GaussianBlur(crop, (41, 21), 0)
        img = imutils.resize(crop, width=8, height=8)
        prediction = classifier.predict(np.array([img.flatten()]))
        print(prediction)

    # Display the resulting frame
    cv2.imshow('FaceDetection', frame)
    cv2.imshow('gray', crop)
    cv2.imshow('thresh', thresh)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 600, 600)
    cv2.imshow('img', img)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
