**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding_windows_1.png
[image4]: ./output_images/sliding_windows_1.png
[image5]: ./output_images/two_cars.png
[image6]: ./output_images/heatmap_1.png
[image7]: ./output_images/heatmap_2.png
[image8]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

Individual steps of this implementation are described below.  

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines # 53 through # 68 of the file called `calculate_features.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `grayscale` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

Please note that for visualization HOG features were calculated on `grayscale` of the image. In actual experiment for vehicle detection `YCrCb` colorspace was used.

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` gave me best results. For colorspaces I mostly played with `RGB` and `YCrCb`. `YCrCb` gave me best result.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `sklearn.SVM`. I used HOG, Spatial and histogram features for training SVM classifier (lines 18-67). Both car and not-car data were first normalized (lines 148-152) and then data was randomly split for training and testing. 80% of data was used for training and 20 % for testing of SVM classifier (lines 157-180). All these steps are in `calculate_features.py` file.

The trained SCM model, along with associated parameters, is stored as pickle file named `svc_pickle.p`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this part is file called `window_fuctions.py`. I used HOG sub-sampling window search as it is more efficient method for doing the sliding window approach. This method (vs traditional sliding window) only needs to extract the Hog features once.

The `find_cars` (lines 15-82) only has to extract hog features once and then can be sub-sampled to get all of its overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. This means that a cells_per_step = 2 would result in a search window overlap of 75%. Its possible to run this same function multiple times for different scale values to generate multiple-scaled search windows.

<!-- ![alt text][image3] -->

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately, I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. I used find cars followed by heatmap (lines 95-109) steps. The code is in file `window_fuctions.py`. Here are some example images:

![alt text][image3]

and the other image is:

![alt text][image4]

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are two frames and their corresponding heatmaps:
First heatmap:

![alt text][image6]

and the second heatmap:

![alt text][image7]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
The link [test video output](./test_car_video_output.mp4)

Link to [project video result](./car_video_output.mp4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline I implemented detects car on the other side of the lane too. I think, its ok but not ideal. Perhaps  a better implementation will detect only "relevant" cars.
