
##Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 
original image
[image1]: test_images/solidWhiteCurve.jpg "Original"
![alt text][image1]
First, I converted the images to grayscale, 
[image2]: test_images_intermidiate/gray_solidWhiteCurve.jpg "Grayscale"
![alt text][image2]
then I i did the gaussian blur smoothing and find edeges using canny
[image3]: test_images_intermidiate/edgessolidWhiteCurve.jpg "Edges"
![alt text][image3]
then i apply the region mask 
[image4]: test_images_intermidiate/maskedsolidWhiteCurve.jpg "Mask and region apply"
![alt text][image4]
Then apply the Hough transform based on choosen parameters
[image5]: test_images_intermidiate/line_imagessolidWhiteCurve.jpg "Hough Transformation apply"
![alt text][image5]
Then apply line image on original using weighs to prepare final images
[image6]: test_images_output/final_solidWhiteCurve.jpg "Final Images"
![alt text][image6]



###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen as per videos if in the region it can detect lines from another lane

Another shortcoming could be to give this area after analysing images of perticular size it is not robust enough.

if there is some truck having lines it will detect lines from back of that also.

if front vehicle is too close this will not detect lines as the lines will be covered.
###3. Suggest possible improvements to your pipeline
In videos we are detecting lot of lines on another lane or the divider. we can eliminate them from our pipeline if we treat them as outlier or their deviation is more than 95% of the lines. 
