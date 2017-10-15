
# Self-Driving Car Engineer Nanodegree


## Project: **Finding Lane Lines on the Road** 
***
In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 

Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.

In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.

---
Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.

**Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**

---

**The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**

---

<figure>
 <img src="line-segments-example.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
 </figcaption>
</figure>
 <p></p> 
<figure>
 <img src="laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
 </figcaption>
</figure>

**Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, see [this forum post](https://carnd-forums.udacity.com/cq/viewquestion.action?spaceKey=CAR&id=29496372&questionTitle=finding-lanes---import-cv2-fails-even-though-python-in-the-terminal-window-has-no-problem-with-import-cv2) for more troubleshooting tips.**  

## Import Packages


```python
%matplotlib inline
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
```

## Read in an Image


```python
#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
```

    This image is: <class 'numpy.ndarray'> with dimensions: (540, 960, 3)





    <matplotlib.image.AxesImage at 0x7f9f7e693cf8>




![png](output_6_2.png)


## Ideas for Lane Detection Pipeline

**Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**

`cv2.inRange()` for color selection  
`cv2.fillPoly()` for regions selection  
`cv2.line()` to draw lines on an image given endpoints  
`cv2.addWeighted()` to coadd / overlay two images
`cv2.cvtColor()` to grayscale or change color
`cv2.imwrite()` to output images to file  
`cv2.bitwise_and()` to apply a mask to an image

**Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

## Helper Functions

Below are some helper functions to help get you started. They should look familiar from the lesson!


```python
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
```

## Test Images

Build your pipeline to work on the images in the directory "test_images"  
**You should make sure your pipeline works well on these images before you try the videos.**


```python
import os
os.listdir("test_images/")
```




    ['solidWhiteRight.jpg',
     'solidYellowCurve.jpg',
     'solidYellowLeft.jpg',
     'solidYellowCurve2.jpg',
     'solidWhiteCurve.jpg',
     'whiteCarLaneSwitch.jpg']



## Build a Lane Finding Pipeline



Build the pipeline and run your solution on all test_images. Make copies into the test_images directory, and you can use the images in your writeup report.

Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.


```python
# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

# Used here on images and below on videos.
def process_image(image , name):
    # you should return the final output (image with lines are drawn on lanes)
    gray = grayscale(image)
    cv2.imwrite('test_images_intermidiate/gray_' + name, gray)
    gaus = gaussian_blur(gray, 3)
    edges = canny(gaus, 50,150)  
    cv2.imwrite('test_images_intermidiate/edges' + name, edges)
    imshape = image.shape
    
    vertices = np.array([[(0,imshape[0]),(450, 310), (490, 310), (imshape[1],imshape[0])]], dtype=np.int32) 
    masked = region_of_interest(edges, vertices)
    cv2.imwrite('test_images_intermidiate/masked' + name, masked)
    
    rho = 2           #distance resolution in pixels of the Hough grid
    theta = 1*np.pi/180  #angular resolution in radians of the Hough grid
    threshold = 50     #minimum number of votes (intersections in Hough grid cell)
    min_line_len = 100  #minimum number of pixels making up a line
    max_line_gap = 120  #maximum gap in pixels between connectable line segments
    line_image = hough_lines(masked, rho, theta, threshold, min_line_len, max_line_gap)
    
    cv2.imwrite('test_images_intermidiate/line_image' + name, line_image)
    result = weighted_img(line_image, image)
    return result



```

### display only using Hough lines


```python
images = os.listdir("test_images/")
for img_file in images:
   
    image = mpimg.imread('test_images/' + img_file)   
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    weighted = process_image(image , img_file)
    f, (ax1) = plt.subplots(1, 1)
    #plt.imshow(weighted)
    ax1.imshow(weighted)
    #plt.imshow(weighted)
    #break-
    weighted = cv2.cvtColor(weighted, cv2.COLOR_BGR2RGB)
    cv2.imwrite('test_images_output/final_' + img_file, weighted)
```


![png](output_18_0.png)



![png](output_18_1.png)



![png](output_18_2.png)



![png](output_18_3.png)



![png](output_18_4.png)



![png](output_18_5.png)


## Test on Videos

You know what's cooler than drawing lanes over images? Drawing lanes over video!

We can test our solution on two provided videos:

`solidWhiteRight.mp4`

`solidYellowLeft.mp4`

**Note: if you get an `import error` when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt. Also, check out [this forum post](https://carnd-forums.udacity.com/questions/22677062/answers/22677109) for more troubleshooting tips.**

**If you get an error that looks like this:**
```
NeedDownloadError: Need ffmpeg exe. 
You can download it by calling: 
imageio.plugins.ffmpeg.download()
```
**Follow the instructions in the error message and check out [this forum post](https://carnd-forums.udacity.com/display/CAR/questions/26218840/import-videofileclip-error) for more troubleshooting tips across operating systems.**


```python
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
```


```python
def skeltonization(binary_line_image_initial ,imshape):

 #skeltonization
    #print(skel)
    imshape = image.shape
    size = np.size(binary_line_image_initial)
    plt.imshow(binary_line_image_initial,'gray')
    skel = np.zeros((imshape[0],imshape[1]), np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    num_itteration=1000
    curr_itter=1
    while( not done):
        eroded = cv2.erode(binary_line_image_initial,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(binary_line_image_initial,temp)
        skel = cv2.bitwise_or(skel,temp)
        binary_line_image_initial = eroded.copy()

        zeros = size - cv2.countNonZero(binary_line_image_initial)
        curr_itter=curr_itter +1
        
        if  num_itteration < curr_itter  :
            done = True
    """ 
    """   
    plt.imshow(skel,'gray')
```

### polynomial fit over hough lines


```python
def process_video_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)

    #image = (mpimg.imread('test_images/solidWhiteCurve.jpg')).astype('uint8')

    gray = grayscale(image)

    # canny(img, low_threshold, high_threshold)
    # gaussian_blur(img, kernel_size)
    # region_of_interest(img, vertices)


    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 1
    blur_gray = gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)   
    ignore_mask_color = 255   

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    #print(imshape)
    vertices = np.array([[(0,imshape[0]),(450, 310), (490, 310), (imshape[1],imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 50     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50 #minimum number of pixels making up a line
    max_line_gap = 100    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

    list_pts=[]
    right_lines = []
    left_lines = []
     # Iterate over the output "lines" and draw lines on a blank image
    for line in lines:
        for x1,y1,x2,y2 in line:
            list_pts.append([x1,y1])
            list_pts.append([x2,y2])
            if(x1 > (image.shape[1]/2) and x2 > (image.shape[1]/2)) :
                right_lines.append([x1,y1])
                right_lines.append([x2,y2])
            elif (x1 < (image.shape[1]/2) and x2 < (image.shape[1]/2)):
                left_lines.append([x1,y1])
                left_lines.append([x2,y2])
            else:
                pass
         
    right_lines = np.array(right_lines )
    left_lines = np.array(left_lines )
    list_pts=np.array(list_pts )
    # Curve fitting approach
    # calculate polynomial fit for the points in right lane
    try:
        right_curve = np.poly1d(np.polyfit(right_lines[:,1], right_lines[:,0], 1))
        left_curve  = np.poly1d(np.polyfit(left_lines[:,1], left_lines[:,0], 1))
        

        max_yValues = np.amin(list_pts[:,1], axis=0)
        #print(max_yValues)

        ly1=max_yValues
        ly2=imshape[0]
        lx1=int(left_curve(ly1))
        lx2=int(left_curve(ly2))

        ry1=imshape[0]
        ry2=max_yValues
        rx1=int(right_curve(ry1))
        rx2=int(right_curve(ry2))

        #print(lx1,ly1)
        #print(lx2,ly2)
        #print(rx1,ry1)
        #print(rx2,ry2)
        cv2.line(line_image,(lx1,ly1),(lx2,ly2),(255,0,0),5)
        cv2.line(line_image,(rx1,ry1),(rx2,ry2),(255,0,0),5)
    
    
    except:
        pass
    

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 

    # Draw the lines on the edge image
    result = weighted_img(image, line_image, α=0.8, β=1., λ=0.)
    #plt.imshow(lines_edges)

    # run your solution on all test_images and make copies into the test_images directory).
    
    
    
    return result
```


```python


images = glob.glob('test_images/*.jpg')
#os.mkdir('test_images_output')
dirname='test_images_output'
for idx, img_file in enumerate(images):
    print(img_file)
    image = mpimg.imread( img_file)   
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    weighted = process_video_image(image )
    
    
    f, (ax1) = plt.subplots(1, 1)
    plt.imshow(weighted)
    #ax1.imshow(weighted)
    weighted = cv2.cvtColor(weighted, cv2.COLOR_BGR2RGB)
    filename=img_file.split('/')[1]
    cv2.imwrite(os.path.join(dirname, 'final_' + filename), weighted)
    
```

    test_images/solidWhiteRight.jpg
    test_images/solidYellowCurve.jpg


    


    test_images/solidYellowLeft.jpg
    test_images/solidYellowCurve2.jpg
    test_images/solidWhiteCurve.jpg
    test_images/whiteCarLaneSwitch.jpg



![png](output_24_3.png)



![png](output_24_4.png)



![png](output_24_5.png)



![png](output_24_6.png)



![png](output_24_7.png)



![png](output_24_8.png)


Let's try the one with the solid white lane on the right first ...


```python
white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_video_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)
```

    [MoviePy] >>>> Building video white.mp4
    [MoviePy] Writing video white.mp4


    100%|█████████▉| 221/222 [00:06<00:00, 32.50it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: white.mp4 
    
    CPU times: user 38.3 s, sys: 1.02 s, total: 39.3 s
    Wall time: 7.28 s


Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.


```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))
```





<video width="960" height="540" controls>
  <source src="white.mp4">
</video>




## Improve the draw_lines() function

**At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**

**Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**

Now for the one with the solid yellow lane on the left. This one's more tricky!


```python
yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_video_image)
%time yellow_clip.write_videofile(yellow_output, audio=False)
```

    [MoviePy] >>>> Building video yellow.mp4
    [MoviePy] Writing video yellow.mp4


    
    
      0%|          | 0/682 [00:00<?, ?it/s][A[A
    
      1%|          | 5/682 [00:00<00:15, 43.41it/s][A[A
    
      1%|▏         | 10/682 [00:00<00:15, 42.42it/s][A[A
    
      2%|▏         | 16/682 [00:00<00:14, 45.92it/s][A[A
    
      3%|▎         | 21/682 [00:00<00:14, 45.14it/s][A[A
    
      4%|▍         | 27/682 [00:00<00:13, 48.07it/s][A[A
    
      5%|▍         | 33/682 [00:00<00:13, 47.99it/s][A[A
    
      6%|▌         | 38/682 [00:00<00:15, 41.29it/s][A[A
    
      6%|▌         | 42/682 [00:00<00:16, 38.43it/s][A[A
    
      7%|▋         | 46/682 [00:01<00:16, 38.45it/s][A[A
    
      7%|▋         | 50/682 [00:01<00:18, 34.68it/s][A[A
    
      8%|▊         | 54/682 [00:01<00:20, 31.08it/s][A[A
    
      9%|▊         | 58/682 [00:01<00:20, 30.55it/s][A[A
    
      9%|▉         | 62/682 [00:01<00:19, 31.72it/s][A[A
    
     10%|▉         | 66/682 [00:01<00:18, 32.77it/s][A[A
    
     10%|█         | 70/682 [00:01<00:20, 30.21it/s][A[A
    
     11%|█         | 74/682 [00:02<00:20, 29.97it/s][A[A
    
     12%|█▏        | 79/682 [00:02<00:18, 31.97it/s][A[A
    
     12%|█▏        | 83/682 [00:02<00:19, 30.14it/s][A[A
    
     13%|█▎        | 87/682 [00:02<00:20, 28.80it/s][A[A
    
     13%|█▎        | 91/682 [00:02<00:19, 30.84it/s][A[A
    
     14%|█▍        | 95/682 [00:02<00:18, 31.51it/s][A[A
    
     15%|█▍        | 99/682 [00:02<00:18, 31.76it/s][A[A
    
     15%|█▌        | 103/682 [00:02<00:19, 29.58it/s][A[A
    
     16%|█▌        | 107/682 [00:03<00:19, 29.43it/s][A[A
    
     16%|█▌        | 110/682 [00:03<00:19, 29.13it/s][A[A
    
     17%|█▋        | 113/682 [00:03<00:21, 26.57it/s][A[A
    
     17%|█▋        | 116/682 [00:03<00:22, 24.94it/s][A[A
    
     18%|█▊        | 120/682 [00:03<00:21, 26.39it/s][A[A
    
     18%|█▊        | 124/682 [00:03<00:19, 28.01it/s][A[A
    
     19%|█▊        | 127/682 [00:03<00:20, 26.58it/s][A[A
    
     19%|█▉        | 130/682 [00:03<00:20, 27.09it/s][A[A
    
     20%|█▉        | 134/682 [00:04<00:18, 29.84it/s][A[A
    
     20%|██        | 138/682 [00:04<00:18, 30.19it/s][A[A
    
     21%|██        | 142/682 [00:04<00:17, 30.99it/s][A[A
    
     21%|██▏       | 146/682 [00:04<00:18, 28.89it/s][A[A
    
     22%|██▏       | 149/682 [00:04<00:20, 26.17it/s][A[A
    
     22%|██▏       | 152/682 [00:04<00:20, 26.31it/s][A[A
    
     23%|██▎       | 155/682 [00:04<00:20, 25.16it/s][A[A
    
     23%|██▎       | 159/682 [00:04<00:18, 27.91it/s][A[A
    
     24%|██▍       | 162/682 [00:05<00:18, 28.17it/s][A[A
    
     24%|██▍       | 165/682 [00:05<00:18, 27.84it/s][A[A
    
     25%|██▍       | 169/682 [00:05<00:17, 29.62it/s][A[A
    
     25%|██▌       | 173/682 [00:05<00:17, 29.80it/s][A[A
    
     26%|██▌       | 177/682 [00:05<00:17, 29.15it/s][A[A
    
     26%|██▋       | 180/682 [00:05<00:18, 27.35it/s][A[A
    
     27%|██▋       | 184/682 [00:05<00:17, 28.43it/s][A[A
    
     27%|██▋       | 187/682 [00:05<00:17, 27.65it/s][A[A
    
     28%|██▊       | 190/682 [00:06<00:17, 28.08it/s][A[A
    
     28%|██▊       | 194/682 [00:06<00:16, 28.90it/s][A[A
    
     29%|██▉       | 197/682 [00:06<00:17, 27.70it/s][A[A
    
     29%|██▉       | 200/682 [00:06<00:18, 26.62it/s][A[A
    
     30%|██▉       | 204/682 [00:06<00:17, 28.10it/s][A[A
    
     30%|███       | 208/682 [00:06<00:15, 30.31it/s][A[A
    
     31%|███       | 212/682 [00:06<00:16, 29.04it/s][A[A
    
     32%|███▏      | 215/682 [00:06<00:16, 28.89it/s][A[A
    
     32%|███▏      | 218/682 [00:07<00:15, 29.08it/s][A[A
    
     32%|███▏      | 221/682 [00:07<00:16, 27.76it/s][A[A
    
     33%|███▎      | 225/682 [00:07<00:15, 29.91it/s][A[A
    
     34%|███▎      | 229/682 [00:07<00:15, 28.95it/s][A[A
    
     34%|███▍      | 232/682 [00:07<00:16, 27.52it/s][A[A
    
     34%|███▍      | 235/682 [00:07<00:17, 26.01it/s][A[A
    
     35%|███▍      | 238/682 [00:07<00:18, 24.59it/s][A[A
    
     35%|███▌      | 242/682 [00:07<00:16, 26.03it/s][A[A
    
     36%|███▌      | 245/682 [00:08<00:16, 25.96it/s][A[A
    
     36%|███▋      | 248/682 [00:08<00:16, 26.69it/s][A[A
    
     37%|███▋      | 251/682 [00:08<00:15, 27.11it/s][A[A
    
     37%|███▋      | 255/682 [00:08<00:15, 28.28it/s][A[A
    
     38%|███▊      | 259/682 [00:08<00:13, 30.79it/s][A[A
    
     39%|███▊      | 263/682 [00:08<00:15, 27.93it/s][A[A
    
     39%|███▉      | 267/682 [00:08<00:14, 29.21it/s][A[A
    
     40%|███▉      | 271/682 [00:08<00:15, 26.42it/s][A[A
    
     40%|████      | 274/682 [00:09<00:15, 25.72it/s][A[A
    
     41%|████      | 277/682 [00:09<00:16, 24.61it/s][A[A
    
     41%|████      | 280/682 [00:09<00:15, 25.82it/s][A[A
    
     42%|████▏     | 285/682 [00:09<00:13, 28.96it/s][A[A
    
     42%|████▏     | 289/682 [00:09<00:13, 28.21it/s][A[A
    
     43%|████▎     | 294/682 [00:09<00:12, 31.20it/s][A[A
    
     44%|████▎     | 298/682 [00:09<00:12, 31.22it/s][A[A
    
     44%|████▍     | 302/682 [00:10<00:13, 28.32it/s][A[A
    
     45%|████▍     | 305/682 [00:10<00:14, 25.75it/s][A[A
    
     45%|████▌     | 308/682 [00:10<00:14, 26.31it/s][A[A
    
     46%|████▌     | 312/682 [00:10<00:13, 28.14it/s][A[A
    
     46%|████▋     | 316/682 [00:10<00:12, 29.42it/s][A[A
    
     47%|████▋     | 320/682 [00:10<00:12, 28.79it/s][A[A
    
     47%|████▋     | 323/682 [00:10<00:13, 27.48it/s][A[A
    
     48%|████▊     | 326/682 [00:10<00:13, 27.27it/s][A[A
    
     48%|████▊     | 329/682 [00:10<00:12, 27.58it/s][A[A
    
     49%|████▊     | 332/682 [00:11<00:13, 26.05it/s][A[A
    
     49%|████▉     | 335/682 [00:11<00:13, 26.18it/s][A[A
    
     50%|████▉     | 338/682 [00:11<00:13, 25.90it/s][A[A
    
     50%|█████     | 341/682 [00:11<00:12, 26.25it/s][A[A
    
     50%|█████     | 344/682 [00:11<00:13, 24.49it/s][A[A
    
     51%|█████     | 348/682 [00:11<00:12, 26.53it/s][A[A
    
     52%|█████▏    | 352/682 [00:11<00:11, 27.59it/s][A[A
    
     52%|█████▏    | 356/682 [00:11<00:11, 28.19it/s][A[A
    
     53%|█████▎    | 359/682 [00:12<00:11, 28.08it/s][A[A
    
     53%|█████▎    | 363/682 [00:12<00:10, 29.46it/s][A[A
    
     54%|█████▍    | 367/682 [00:12<00:10, 30.13it/s][A[A
    
     54%|█████▍    | 371/682 [00:12<00:09, 31.71it/s][A[A
    
     55%|█████▍    | 375/682 [00:12<00:10, 30.37it/s][A[A
    
     56%|█████▌    | 379/682 [00:12<00:10, 29.83it/s][A[A
    
     56%|█████▌    | 383/682 [00:12<00:10, 28.80it/s][A[A
    
     57%|█████▋    | 387/682 [00:13<00:10, 29.02it/s][A[A
    
     57%|█████▋    | 390/682 [00:13<00:11, 26.11it/s][A[A
    
     58%|█████▊    | 394/682 [00:13<00:10, 27.68it/s][A[A
    
     58%|█████▊    | 398/682 [00:13<00:10, 27.71it/s][A[A
    
     59%|█████▉    | 402/682 [00:13<00:09, 28.82it/s][A[A
    
     60%|█████▉    | 406/682 [00:13<00:09, 29.42it/s][A[A
    
     60%|█████▉    | 409/682 [00:13<00:09, 28.47it/s][A[A
    
     61%|██████    | 413/682 [00:13<00:08, 30.54it/s][A[A
    
     61%|██████    | 417/682 [00:14<00:08, 30.89it/s][A[A
    
     62%|██████▏   | 421/682 [00:14<00:08, 31.07it/s][A[A
    
     63%|██████▎   | 427/682 [00:14<00:07, 33.98it/s][A[A
    
     63%|██████▎   | 431/682 [00:14<00:07, 32.71it/s][A[A
    
     64%|██████▍   | 435/682 [00:14<00:07, 33.74it/s][A[A
    
     65%|██████▍   | 440/682 [00:14<00:06, 36.92it/s][A[A
    
     65%|██████▌   | 444/682 [00:14<00:07, 33.66it/s][A[A
    
     66%|██████▌   | 448/682 [00:14<00:07, 31.48it/s][A[A
    
     66%|██████▋   | 452/682 [00:15<00:07, 29.59it/s][A[A
    
     67%|██████▋   | 456/682 [00:15<00:07, 29.39it/s][A[A
    
     67%|██████▋   | 460/682 [00:15<00:07, 28.20it/s][A[A
    
     68%|██████▊   | 463/682 [00:15<00:08, 26.73it/s][A[A
    
     68%|██████▊   | 467/682 [00:15<00:07, 28.57it/s][A[A
    
     69%|██████▉   | 471/682 [00:15<00:07, 28.71it/s][A[A
    
     70%|██████▉   | 474/682 [00:15<00:07, 28.77it/s][A[A
    
     70%|██████▉   | 477/682 [00:15<00:07, 29.10it/s][A[A
    
     71%|███████   | 481/682 [00:16<00:06, 31.66it/s][A[A
    
     71%|███████▏  | 486/682 [00:16<00:05, 34.70it/s][A[A
    
     72%|███████▏  | 490/682 [00:16<00:05, 34.70it/s][A[A
    
     72%|███████▏  | 494/682 [00:16<00:06, 31.08it/s][A[A
    
     73%|███████▎  | 498/682 [00:16<00:05, 30.91it/s][A[A
    
     74%|███████▎  | 502/682 [00:16<00:06, 28.43it/s][A[A
    
     74%|███████▍  | 506/682 [00:16<00:06, 28.58it/s][A[A
    
     75%|███████▍  | 509/682 [00:17<00:06, 28.35it/s][A[A
    
     75%|███████▌  | 512/682 [00:17<00:06, 26.98it/s][A[A
    
     76%|███████▌  | 515/682 [00:17<00:06, 26.65it/s][A[A
    
     76%|███████▌  | 518/682 [00:17<00:06, 26.67it/s][A[A
    
     77%|███████▋  | 522/682 [00:17<00:05, 28.87it/s][A[A
    
     77%|███████▋  | 526/682 [00:17<00:05, 30.05it/s][A[A
    
     78%|███████▊  | 530/682 [00:17<00:04, 31.30it/s][A[A
    
     78%|███████▊  | 534/682 [00:17<00:04, 29.66it/s][A[A
    
     79%|███████▉  | 538/682 [00:17<00:04, 30.26it/s][A[A
    
     79%|███████▉  | 542/682 [00:18<00:04, 28.95it/s][A[A
    
     80%|████████  | 546/682 [00:18<00:04, 29.47it/s][A[A
    
     80%|████████  | 549/682 [00:18<00:04, 28.78it/s][A[A
    
     81%|████████  | 553/682 [00:18<00:04, 30.46it/s][A[A
    
     82%|████████▏ | 557/682 [00:18<00:03, 31.59it/s][A[A
    
     82%|████████▏ | 561/682 [00:18<00:03, 33.41it/s][A[A
    
     83%|████████▎ | 565/682 [00:18<00:03, 29.65it/s][A[A
    
     83%|████████▎ | 569/682 [00:19<00:03, 28.89it/s][A[A
    
     84%|████████▍ | 572/682 [00:19<00:03, 29.21it/s][A[A
    
     84%|████████▍ | 575/682 [00:19<00:03, 28.00it/s][A[A
    
     85%|████████▍ | 578/682 [00:19<00:03, 28.40it/s][A[A
    
     85%|████████▌ | 582/682 [00:19<00:03, 30.35it/s][A[A
    
     86%|████████▌ | 586/682 [00:19<00:03, 29.20it/s][A[A
    
     87%|████████▋ | 590/682 [00:19<00:02, 31.60it/s][A[A
    
     87%|████████▋ | 594/682 [00:19<00:02, 29.74it/s][A[A
    
     88%|████████▊ | 598/682 [00:19<00:02, 31.90it/s][A[A
    
     88%|████████▊ | 602/682 [00:20<00:02, 32.10it/s][A[A
    
     89%|████████▉ | 606/682 [00:20<00:02, 32.70it/s][A[A
    
     89%|████████▉ | 610/682 [00:20<00:02, 29.81it/s][A[A
    
     90%|█████████ | 614/682 [00:20<00:02, 29.07it/s][A[A
    
     91%|█████████ | 618/682 [00:20<00:02, 29.68it/s][A[A
    
     91%|█████████▏| 623/682 [00:20<00:01, 33.01it/s][A[A
    
     92%|█████████▏| 627/682 [00:20<00:01, 33.10it/s][A[A
    
     93%|█████████▎| 631/682 [00:21<00:01, 28.90it/s][A[A
    
     93%|█████████▎| 635/682 [00:21<00:01, 30.18it/s][A[A
    
     94%|█████████▎| 639/682 [00:21<00:01, 29.56it/s][A[A
    
     94%|█████████▍| 643/682 [00:21<00:01, 28.26it/s][A[A
    
     95%|█████████▍| 647/682 [00:21<00:01, 30.75it/s][A[A
    
     95%|█████████▌| 651/682 [00:21<00:00, 31.13it/s][A[A
    
     96%|█████████▌| 655/682 [00:21<00:00, 31.99it/s][A[A
    
     97%|█████████▋| 659/682 [00:21<00:00, 30.13it/s][A[A
    
     97%|█████████▋| 663/682 [00:22<00:00, 30.71it/s][A[A
    
     98%|█████████▊| 668/682 [00:22<00:00, 33.59it/s][A[A
    
     99%|█████████▊| 672/682 [00:22<00:00, 31.61it/s][A[A
    
     99%|█████████▉| 677/682 [00:22<00:00, 33.00it/s][A[A
    
    100%|█████████▉| 681/682 [00:22<00:00, 34.12it/s][A[A
    
    [A[A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: yellow.mp4 
    
    CPU times: user 2min 1s, sys: 3.81 s, total: 2min 5s
    Wall time: 23.1 s



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))
```





<video width="960" height="540" controls>
  <source src="yellow.mp4">
</video>




## Writeup and Submission

If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.


## Optional Challenge

Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!


```python
challenge_output = 'extra.mp4'
clip2 = VideoFileClip('challenge.mp4')
challenge_clip = clip2.fl_image(process_video_image)
%time challenge_clip.write_videofile(challenge_output, audio=False)
```

    [MoviePy] >>>> Building video extra.mp4
    [MoviePy] Writing video extra.mp4


    
      0%|          | 0/251 [00:00<?, ?it/s][A
      1%|          | 3/251 [00:00<00:10, 23.93it/s][A
      2%|▏         | 6/251 [00:00<00:10, 23.52it/s][A
      4%|▎         | 9/251 [00:00<00:10, 23.54it/s][A
      5%|▍         | 12/251 [00:00<00:09, 23.91it/s][A
      6%|▌         | 15/251 [00:00<00:09, 23.63it/s][A
      7%|▋         | 18/251 [00:00<00:09, 23.92it/s][A
      8%|▊         | 21/251 [00:00<00:09, 24.49it/s][A
     10%|▉         | 24/251 [00:01<00:09, 23.60it/s][A
     11%|█         | 27/251 [00:01<00:09, 24.13it/s][A
     12%|█▏        | 30/251 [00:01<00:09, 24.38it/s][A
     13%|█▎        | 33/251 [00:01<00:09, 23.53it/s][A
     14%|█▍        | 36/251 [00:01<00:09, 22.85it/s][A
     16%|█▌        | 39/251 [00:01<00:09, 22.79it/s][A
     17%|█▋        | 42/251 [00:01<00:09, 22.18it/s][A
     18%|█▊        | 45/251 [00:01<00:10, 20.60it/s][A
     19%|█▉        | 48/251 [00:02<00:09, 20.37it/s][A
     20%|██        | 51/251 [00:02<00:10, 19.29it/s][A
     21%|██        | 53/251 [00:02<00:10, 19.32it/s][A
     22%|██▏       | 55/251 [00:02<00:10, 17.85it/s][A
     23%|██▎       | 58/251 [00:02<00:10, 18.44it/s][A
     24%|██▍       | 61/251 [00:02<00:09, 19.03it/s][A
     25%|██▌       | 63/251 [00:02<00:10, 17.39it/s][A
     26%|██▌       | 65/251 [00:03<00:10, 17.18it/s][A
     27%|██▋       | 67/251 [00:03<00:10, 17.52it/s][A
     27%|██▋       | 69/251 [00:03<00:10, 17.85it/s][A
     28%|██▊       | 71/251 [00:03<00:10, 17.41it/s][A
     29%|██▉       | 73/251 [00:03<00:10, 16.40it/s][A
     30%|██▉       | 75/251 [00:03<00:10, 16.00it/s][A
     31%|███       | 77/251 [00:03<00:11, 15.51it/s][A
     31%|███▏      | 79/251 [00:03<00:11, 15.00it/s][A
     32%|███▏      | 81/251 [00:04<00:11, 15.32it/s][A
     33%|███▎      | 83/251 [00:04<00:10, 16.10it/s][A
     34%|███▍      | 85/251 [00:04<00:10, 16.55it/s][A
     35%|███▍      | 87/251 [00:04<00:10, 15.63it/s][A
     35%|███▌      | 89/251 [00:04<00:10, 15.43it/s][A
     36%|███▋      | 91/251 [00:04<00:10, 14.78it/s][A
     37%|███▋      | 93/251 [00:04<00:10, 15.15it/s][A
     38%|███▊      | 95/251 [00:05<00:10, 14.80it/s][A
     39%|███▊      | 97/251 [00:05<00:10, 14.70it/s][A
     39%|███▉      | 99/251 [00:05<00:10, 14.38it/s][A
     40%|████      | 101/251 [00:05<00:10, 14.59it/s][A
     41%|████      | 103/251 [00:05<00:09, 15.06it/s][A
     42%|████▏     | 105/251 [00:05<00:09, 15.93it/s][A
     43%|████▎     | 107/251 [00:05<00:08, 16.85it/s][A
     43%|████▎     | 109/251 [00:05<00:08, 16.22it/s][A
     44%|████▍     | 111/251 [00:06<00:08, 16.69it/s][A
     45%|████▌     | 113/251 [00:06<00:08, 16.69it/s][A
     46%|████▌     | 115/251 [00:06<00:08, 16.46it/s][A
     47%|████▋     | 117/251 [00:06<00:07, 17.19it/s][A
     47%|████▋     | 119/251 [00:06<00:07, 16.59it/s][A
     48%|████▊     | 121/251 [00:06<00:07, 16.26it/s][A
     49%|████▉     | 123/251 [00:06<00:07, 16.83it/s][A
     50%|█████     | 126/251 [00:06<00:07, 17.64it/s][A
     51%|█████     | 128/251 [00:06<00:06, 18.16it/s][A
     52%|█████▏    | 130/251 [00:07<00:06, 18.55it/s][A
     53%|█████▎    | 132/251 [00:07<00:06, 18.83it/s][A
     53%|█████▎    | 134/251 [00:07<00:06, 16.73it/s][A
     54%|█████▍    | 136/251 [00:07<00:07, 16.03it/s][A
     55%|█████▍    | 138/251 [00:07<00:07, 15.75it/s][A
     56%|█████▌    | 140/251 [00:07<00:06, 16.57it/s][A
     57%|█████▋    | 142/251 [00:07<00:06, 16.86it/s][A
     57%|█████▋    | 144/251 [00:07<00:06, 17.12it/s][A
     58%|█████▊    | 146/251 [00:08<00:06, 16.24it/s][A
     59%|█████▉    | 148/251 [00:08<00:06, 16.08it/s][A
     60%|█████▉    | 150/251 [00:08<00:06, 15.84it/s][A
     61%|██████    | 152/251 [00:08<00:05, 16.88it/s][A
     61%|██████▏   | 154/251 [00:08<00:05, 16.74it/s][A
     63%|██████▎   | 157/251 [00:08<00:05, 17.71it/s][A
     64%|██████▎   | 160/251 [00:08<00:04, 18.56it/s][A
     65%|██████▍   | 162/251 [00:08<00:04, 17.91it/s][A
     65%|██████▌   | 164/251 [00:09<00:04, 17.96it/s][A
     66%|██████▌   | 166/251 [00:09<00:04, 17.19it/s][A
     67%|██████▋   | 168/251 [00:09<00:04, 17.46it/s][A
     68%|██████▊   | 170/251 [00:09<00:04, 16.91it/s][A
     69%|██████▊   | 172/251 [00:09<00:04, 16.84it/s][A
     69%|██████▉   | 174/251 [00:09<00:04, 16.35it/s][A
     70%|███████   | 176/251 [00:09<00:04, 16.42it/s][A
     71%|███████   | 178/251 [00:09<00:04, 16.72it/s][A
     72%|███████▏  | 180/251 [00:10<00:04, 16.67it/s][A
     73%|███████▎  | 182/251 [00:10<00:04, 16.50it/s][A
     73%|███████▎  | 184/251 [00:10<00:03, 17.22it/s][A
     74%|███████▍  | 186/251 [00:10<00:03, 17.23it/s][A
     75%|███████▍  | 188/251 [00:10<00:03, 16.53it/s][A
     76%|███████▌  | 190/251 [00:10<00:03, 17.32it/s][A
     76%|███████▋  | 192/251 [00:10<00:03, 16.66it/s][A
     77%|███████▋  | 194/251 [00:10<00:03, 16.44it/s][A
     78%|███████▊  | 196/251 [00:11<00:03, 15.82it/s][A
     79%|███████▉  | 198/251 [00:11<00:03, 16.77it/s][A
     80%|███████▉  | 200/251 [00:11<00:03, 16.75it/s][A
     81%|████████  | 203/251 [00:11<00:02, 17.71it/s][A
     82%|████████▏ | 205/251 [00:11<00:02, 16.43it/s][A
     82%|████████▏ | 207/251 [00:11<00:02, 16.61it/s][A
     83%|████████▎ | 209/251 [00:11<00:02, 16.58it/s][A
     84%|████████▍ | 211/251 [00:11<00:02, 16.76it/s][A
     85%|████████▍ | 213/251 [00:12<00:02, 17.03it/s][A
     86%|████████▌ | 215/251 [00:12<00:02, 17.30it/s][A
     86%|████████▋ | 217/251 [00:12<00:01, 17.22it/s][A
     87%|████████▋ | 219/251 [00:12<00:01, 17.80it/s][A
     88%|████████▊ | 222/251 [00:12<00:01, 17.79it/s][A
     89%|████████▉ | 224/251 [00:12<00:01, 18.38it/s][A
     90%|█████████ | 226/251 [00:12<00:01, 14.57it/s][A
     91%|█████████ | 228/251 [00:13<00:02, 10.31it/s][A
     92%|█████████▏| 230/251 [00:13<00:02,  8.99it/s][A
     92%|█████████▏| 232/251 [00:13<00:02,  8.26it/s][A
     93%|█████████▎| 234/251 [00:13<00:01,  9.30it/s][A
     94%|█████████▍| 236/251 [00:14<00:01, 10.40it/s][A
     95%|█████████▍| 238/251 [00:14<00:01, 12.01it/s][A
     96%|█████████▌| 240/251 [00:14<00:00, 13.26it/s][A
     96%|█████████▋| 242/251 [00:14<00:00, 14.01it/s][A
     97%|█████████▋| 244/251 [00:14<00:00, 14.67it/s][A
     98%|█████████▊| 246/251 [00:14<00:00, 15.39it/s][A
     99%|█████████▉| 248/251 [00:14<00:00, 16.52it/s][A
    100%|█████████▉| 250/251 [00:14<00:00, 15.64it/s][A
    100%|██████████| 251/251 [00:14<00:00, 16.87it/s][A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: extra.mp4 
    
    CPU times: user 1min 3s, sys: 2.96 s, total: 1min 6s
    Wall time: 16.1 s



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))
```





<video width="960" height="540" controls>
  <source src="extra.mp4">
</video>




#**Finding Lane Lines on the Road** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image0]: test_images/solidWhiteCurve.jpg "Grayscale"
![alt text][image0]
---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 
original image
[image1]: test_images/solidWhiteCurve.jpg?31312 "Original"
![alt text][image1]
First, I converted the images to grayscale, 
[image2]: test_images_intermidiate/gray_solidWhiteCurve.jpg?3112 "Grayscale"
![alt text][image2]
then I i did the gaussian blur smoothing and find edeges using canny
[image3]: test_images_intermidiate/edgessolidWhiteCurve.jpg?1312 "Edges"
![alt text][image3]
then i apply the region mask 
[image4]: test_images_intermidiate/maskedsolidWhiteCurve.jpg?312 "Mask and region apply"
![alt text][image4]
Then apply the Hough transform based on choosen parameters
[image5]: test_images_intermidiate/line_imagesolidWhiteCurve.jpg?2332 "Hough Transformation apply"
![alt text][image5]
Then apply line image on original using weights to prepare final images
[image6]: test_images_output/final_solidWhiteCurve.jpg?arg12 "Final Images"
![alt text][image6]



###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen as per videos if in the region it can detect lines from another lane

Another shortcoming could be to give this area after analysing images of perticular size it is not robust enough.

if there is some truck having lines it will detect lines from back of that also.

if front vehicle is too close this will not detect lines as the lines will be covered.
###3. Suggest possible improvements to your pipeline
In videos we are detecting lot of lines on another lane or the divider. we can eliminate them from our pipeline if we treat them as outlier or their deviation is more than 95% of the lines. 


```python

```
