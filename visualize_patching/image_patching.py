import numpy as np 
import matplotlib.pyplot as plt 
import cv2

def divide(img,m,n):
    h, w = img.shape[0],img.shape[1]
    grid_h=int(h*1.0/(m-1)+0.5)
    grid_w=int(w*1.0/(n-1)+0.5)
    h=grid_h*(m-1)
    w=grid_w*(n-1)
    img_re=cv2.resize(img,(w,h),cv2.INTER_LINEAR)
    gx, gy = np.meshgrid(np.linspace(0, w, n),np.linspace(0, h, m))
    gx=gx.astype(np.int)
    gy=gy.astype(np.int)
    divide_image = np.zeros([m-1, n-1, grid_h, grid_w,3], np.uint8)
    for i in range(m-1):
        for j in range(n-1):      
            divide_image[i,j,...]=img_re[
            gy[i][j]:gy[i+1][j+1], gx[i][j]:gx[i+1][j+1],:]
    return divide_image

def display_blocks(divide_image):#    
    m,n=divide_image.shape[0],divide_image.shape[1]
    for i in range(m):
        for j in range(n):
            plt.subplot(m,n,i*n+j+1)
            plt.imshow(divide_image[i,j,:])
            plt.axis('off')
    plt.show()

if __name__ == '__main__':

   img = cv2.imread('generate.png')
   img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
   h, w = img.shape[0], img.shape[1]
   fig1 = plt.figure('Image')
   cv2.imshow(winname = "img", mat = img)
   plt.axis('off')
   plt.title('Original image')
   divide_img = divide(img,9,2)
   display_blocks(divide_img)
