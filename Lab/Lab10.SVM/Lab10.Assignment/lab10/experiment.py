import cv2
from train import processFiles, trainSVM
from detector import Detector

import os
print("当前路径:", os.getcwd())

# Replace these with the directories containing your
# positive and negative sample images, respectively.
path="Lab10.SVM/Lab10.Assignment/"
pos_dir = path+"samples/vehicles"
neg_dir = path+"samples/non-vehicles"

# Replace this with the path to your test video file.
video_file = path+"videos/test_video.mp4"


def experiment1():
    """
    Train a classifier and run it on a video using default settings
    without saving any files to disk.

    Validation Phase.

    Validation Accuracy:  0.9704225352112676
    Validation Precision:  0.9721274175199089
    Validation Recall:  0.968271954674221
    Validation F-1 Score:  0.9701958558047119
    Testing Phase.

    Testing Accuracy:  0.9740990990990991
    Testing Precision:  0.9886363636363636
    Testing Recall:  0.9602649006622517
    Testing F-1 Score:  0.9742441209406494
    """
    # TODO: You need to adjust hyperparameters
    # 从样本中提取HOG特征，并将结果及参数封装到一个字典中返回
    # Extract HOG features from images in the sample directories and 
    # return results and parameters in a dict.
    feature_data = processFiles(pos_dir, neg_dir, recurse=True,hog_features=True)


    # Train SVM and return the classifier and parameters in a dict.
    # This function takes the dict from processFiles() as an input arg.
    classifier_data = trainSVM(feature_data=feature_data)


    # TODO: You need to adjust hyperparameters of loadClassifier() and detectVideo()
    #       to obtain a better performance

    # Instantiate a Detector object and load the dict from trainSVM().
    detector = Detector().loadClassifier(classifier_data=classifier_data)
  
    # Open a VideoCapture object for the video file.
    cap = cv2.VideoCapture(video_file)

    # Start the detector by supplying it with the VideoCapture object.
    # At this point, the video will be displayed, with bounding boxes
    # drawn around detected objects per the method detailed in README.md.
    detector.detectVideo(video_capture=cap)

    """
    color_sapce = "HSV" 
        Validation Phase.

        Validation Accuracy:  0.9312676056338028
        Validation Precision:  0.9618885096700797
        Validation Recall:  0.9052462526766595
        Validation F-1 Score:  0.9327082184225041
        Testing Phase.

        Testing Accuracy:  0.9369369369369369
        Testing Precision:  0.9613636363636363
        Testing Recall:  0.9155844155844156
        Testing F-1 Score:  0.9379157427937915

        C 调大，准确率略微提升
        loss 改为hinge 效果不好
        fit_intercept true 效果提升

        Validation Phase.

        Validation Accuracy:  0.9715492957746479
        Validation Precision:  0.9766780432309442
        Validation Recall:  0.9662352279122116
        Validation F-1 Score:  0.9714285714285715

        Testing Phase.

        Testing Accuracy:  0.9774774774774775
        Testing Precision:  0.9840909090909091
        Testing Recall:  0.9708520179372198
        Testing F-1 Score:  0.9774266365688489

        C=1, loss=squared_hinge, penalty=l2, dual=True, fit_intercept=True

    """
    """
    pix_per_cell=(4,4),cells_per_block=(2,2)
    能选上白车
    8,4 no
    16 4
    """
def experiment2():
    feature_data = processFiles(pos_dir, neg_dir, recurse=True,hog_features=True,pix_per_cell=(4,4),cells_per_block=(2,2))
    # C=1, loss=squared_hinge, penalty=l2, dual=True, fit_intercept=True
    classifier_data = trainSVM(feature_data=feature_data,dual=True,fit_intercept=True,C=1)
    # scale=1.5,y_step=0.05
    detector = Detector(y_range=(0.2,1)).loadClassifier(classifier_data=classifier_data)
    cap = cv2.VideoCapture(video_file)
    detector.detectVideo(video_capture=cap)

if __name__ == "__main__":
    # experiment1()
    experiment2() # may you need to try other parameters
    # experiment3 ...


