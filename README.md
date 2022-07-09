# Panorama-Image
Implementation of Panorama algorithm using Computer Vision methods

# Program Description
The project includes the implementation of the following:
1. Projective Homography Calculation
2. Forward Homography:
   - Slow method
   - Fast method
3. Test Homography by MSE and fit_percentage
4. RANSAC Algorithm 
5. Backward Mapping Calculation
6. Panorama Algorithm 


## Projective Transformation Equations
Building a system of equations of the form 𝐴𝑥 = 𝑏, for projective transformation as follow:
<br /> 
![image](https://user-images.githubusercontent.com/108329249/178119974-738f4356-9dca-4ba4-a025-b6685c1cf340.png)
<br /> 
When X is the source coordinate and v',u' are the destination coordinate.<br /> 
You should use at least 4 pair of points in order to solve 8 degrees of freedom.<br /> 
The solution of the system is the eigenvector corresponding to the minimal eigenvalue.<br /> 
I used SVD decomposition in order to find the solution.  

## RANSAC Algorithm
w -> percentage of inliers (in our case 80%)<br />
p -> probability of success (in our case 99%)<br />
n -> minimal pairs of point to calculate a model (in our case 4)<br />
k –> number of iteration of RANSAC algorithm<br />
Assuming that:<br />
![image](https://user-images.githubusercontent.com/108329249/178119672-89eef095-c00a-4f05-ba6b-ed39ac9e86f9.png)
<br />
We rounded k and increase by +1 to avoid the case where w=1
<br />
k = ⌈k⌉ + 1 = 10 
<br />
<br />
Loop for k iteration:
  - Sample n pairs of points randomly.
  - Compute the model (homography) using the random points.
  - Find inliers (all the points with distance < t) using the calculated homography.
  - If number of inliers > d:
    - Recompute the model (homography) using all inliers.
    - If error < previous error:
      - Save the model (homography).
      
Return the best model (homography).
### RANSAC Forward
![image](https://user-images.githubusercontent.com/108329249/178119817-20dcdaea-168c-4538-b179-a99fcd58199e.png)

### RANSAC Backward
First calculated the backward homography using RANSAC, then applied it and finally computed bilinear interpolation.
<br />
![image](https://user-images.githubusercontent.com/108329249/178120072-79fb4dee-95cd-4e8c-ad60-dd43a27cf81a.png)
<br />

![image](https://user-images.githubusercontent.com/108329249/178119838-12b06320-e928-4436-9117-65fc62565a18.png)

## Panorama Algorithm
- Computing forward homography using RANSAC.
- Finding panorama shape (using the forward homography).
- Calculate backward homography by inverse and normalization of the forward homography<br />
![image](https://user-images.githubusercontent.com/108329249/178120079-7cf753e3-fbed-4461-a160-489abe524642.png)
- Apply translation on the backward homography to get the panorama homography.
- Create panorama clear canvas.
- Perform backward warping of the source image using the panorama homography.
- Plant the destination image in the panorama canvas.
- Place the backward warped image into the panorama “free” space.
- Return the panorama image.

![image](https://user-images.githubusercontent.com/108329249/178120139-28e91fa6-12b9-4e6d-aa05-6af57b1e0eee.png)











