import cv2
import numpy as np
from deepface import DeepFace


#------------------------Q1--------------------------
#first converted the iamge to gray scale then applied thresholding then found 
#rectabgles using contours checked for polygons and then centroid and marked the white recatnge with red outline
def Question1_WhiteRectDetection(imagePath):

    image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    grayScaleImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(grayScaleImg, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
      epsilon = 0.04 * cv2.arcLength(contour, True)
      approx = cv2.approxPolyDP(contour, epsilon, True)

    
    if len(approx) == 4:
        perRect = cv2.arcLength(contour, True)
        M = cv2.moments(contour)
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])

        cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)
        whiteRectDetected = True

        print(f"Parameter of the rectangle: {perRect}")
        print(f"Centroid of the rectangle: ({centroid_x}, {centroid_y})")


    if(whiteRectDetected):
      print("Yes! White rectangle has been detcted")

    else:
      print("NO! there is not white rectangle")

    # Display the image with contours
    cv2.imshow("Image with Rectangle", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




#-----------------------------Q2-----------------------------
#wrote function to remove background of the image via creating mask for white colour then seperated the face via inverting the mask
def removeBackground(imagePath):
  
  image = cv2.imread(imagePath)
  lower_white = np.array([200, 200, 200])
  upper_white = np.array([255, 255, 255])

  mask = cv2.inRange(image, lower_white, upper_white)
  mask = cv2.bitwise_not(mask)

  result = cv2.bitwise_and(image, image, mask=mask)
  outputFile="face_only1.png"

  cv2.imwrite(outputFile, result)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  return outputFile

#first found the length of the face via finding largest contour then found the length of the hair with same technique
#then if hair lenght is equal or greater to face length distinguished as female and rest as male
def Question2_DetectGender(imagePathBg, imagePathNoBg):
   
   #finding face length
   faceImgNoBg = cv2.imread(imagePathNoBg)
   grayFace = cv2.cvtColor(faceImgNoBg, cv2.COLOR_BGR2GRAY)
   contours, _ = cv2.findContours(grayFace, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   largest_contour = max(contours, key=cv2.contourArea)
   y_coordinates = [point[0][1] for point in largest_contour]
   faceLengthY = max(y_coordinates) - min(y_coordinates)
   print("Length of the face:", faceLengthY)


   #finding hair length
   faceImgBg = cv2.imread(imagePathBg)
   hsv_image = cv2.cvtColor(faceImgBg, cv2.COLOR_BGR2HSV)
   lower_black = np.array([0, 0, 0], dtype=np.uint8)
   upper_black = np.array([180, 255, 30], dtype=np.uint8)
   black_mask = cv2.inRange(hsv_image, lower_black, upper_black)
   contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   maxYCordHair = 0

   for contour in contours:
     maxContY = max(point[0][1] for point in contour)
    
     if maxContY > maxYCordHair:
        maxYCordHair = maxContY

     print("Maximum length of black hair :", maxYCordHair)

     if(maxYCordHair >=faceLengthY):
        return "Female"
     else:
        return "Male"


    


#--------------------------Q3---------------------------------
#converted image to greyScale
def grayScaleImage(inputPath):
    outputPath="grayImg.png"
    image = cv2.imread(inputPath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(outputPath, gray)  
    return outputPath

#blurred the image
def blurImage(inputPath, kernel_size=(5, 5)):
    outputPath="blurImage.png"
    image = cv2.imread(inputPath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, kernel_size, 0)
    cv2.imwrite(outputPath, blurred)  
    return outputPath

#found the laplacian variance of both images then compared for both one with less var was blurred and the other was original
def Question3_FindBlurAndGreyImage(image1, image2, threshold=100):

    image1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

    laplacian_var1 = cv2.Laplacian(image1, cv2.CV_64F).var()
    laplacian_var2 = cv2.Laplacian(image2, cv2.CV_64F).var()

    if laplacian_var1 < laplacian_var2:
        blurred_image = image1
        original_image = image2
    else:
        blurred_image = image2
        original_image = image1

    cv2.imshow("Blurred Image", blurred_image)
    cv2.imshow("Original Image", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   

#-------------------------Q4-----------------------------------
#found area of the speicifc colours via suming the pixels using numpy and then foudn centroid 
# of that colour using mean of cols and rows of that col and placed the area on the position on the image
def Question4_FindArea(imagePath):
  
  image = cv2.imread(imagePath)

  uniqueColours = ['#fed966', '#e7e5e6', '#d9d9d9', '#aeaaa9']

  for tgtColHex in uniqueColours:
    colourRgb = np.array([int(tgtColHex[i:i+2], 16) for i in (1, 3, 5)])
    colourBgr = colourRgb[::-1]  

    maskColour = np.all(image == colourBgr, axis=-1)

    area = np.sum(maskColour)

    if area > 0:
        print(f"The color {tgtColHex} exists in the image.")
        print(f"Area of color {tgtColHex}: {area} pixels")

        rows, cols = np.where(maskColour)
        centroid_x = int(np.mean(cols))
        centroid_y = int(np.mean(rows))
        print(f"Centroid of color {tgtColHex}: (x={centroid_x}, y={centroid_y})")

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, f"{area}", (centroid_x, centroid_y), font, 0.3, (0, 0, 255), 1)
        
    else:
        print(f"The color {tgtColHex} does not exist in the image.")

  cv2.imwrite('fig1Area.jpg', image)
  cv2.imshow('Image with Area', image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

#--------------------------------Q5----------------------------
#found the width of each bar on x axis then found the number of red pixels within that width on x axis
def Question5_FindArrowPercent(imagePath):
    image = cv2.imread(imagePath)

    bgrColValues = {
        'Yellow': (102, 217, 254),
        'Light Gray': (230, 229, 231),
        'Gray': (217, 217, 217),
        'Dark Gray': (169, 170, 174),
    }

    results = {}
    prevXpoint = 0
    prev_color = ""

    for colName, color_bgr in bgrColValues.items():

        color_mask = np.all(image == color_bgr, axis=-1)
        latestXpoint = np.max(np.where(np.any(color_mask, axis=0)))

        #finding width of the bar
        barWidth = latestXpoint - prevXpoint

        #finding percentage of red pixel within the width of bar
        redMask = np.all(image[:, prevXpoint:latestXpoint] == (36, 27, 237), axis=-1)
        redArrowPixelCount = np.sum(redMask)
        totalPixelImage = barWidth * image.shape[0]
        redPixelPercent = (redArrowPixelCount / totalPixelImage) * 100

        results[colName] = {
            'Bar Width': barWidth,
            'Red Pixel Count': redArrowPixelCount,
            'Percentage Red Pixels': redPixelPercent,
        }

        prevXpoint = latestXpoint
        prev_color = colName

    for colName, data in results.items():
        print(f'{colName}:')
        print(f'  Bar Width: {data["Bar Width"]} pixels')
        print(f'  Red Pixel Count: {data["Red Pixel Count"]}')
        print(f'  Percentage Red Pixels: {data["Percentage Red Pixels"]:.2f}%')
        print()

  
#----------------------------Q6--------------------------------
#sepearted the desired pixels of specific colour with some tolerance to handle variations in shade of colour
#then used bitwise AND to segment that area and save it in sepearte file
def Question6_SegmentBones(imagePath, findColour, tolerance, outputPath):
    image = cv2.imread(imagePath)

    lower_bound = np.array([max(0, c - tolerance) for c in findColour], dtype=np.uint8)
    upper_bound = np.array([min(255, c + tolerance) for c in findColour], dtype=np.uint8)
    color_mask = cv2.inRange(image, lower_bound, upper_bound)

    segmentedBone = cv2.bitwise_and(image, image, mask=color_mask)

    cv2.imwrite(outputPath, segmentedBone)

    max_width = segmentedBone.shape[1]
    max_height = segmentedBone.shape[0]

    print(f"Maximum Width: {max_width} pixels")
    print(f"Maximum Height: {max_height} pixels")

    cv2.imshow('Segmented Bone Area', segmentedBone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#---------------------------main-------------------------------
#Run Q1-------------------------------------------------------
imagePath = "rect1.jpg" 
# Question1_WhiteRectDetection(imagePath)

#Run Q2------------------------------------------------------
Q2Image1Path="fig3.jpg"
Q2Image2Path="fig4.jpg"

# Q2Image1NoBg=removeBackground(Q2Image1Path)
# Q2Image1Gender=Question2_DetectGender(Q2Image1Path, Q2Image1NoBg)
# print(f"the gender of image is {Q2Image1Gender}")

# Q2Image2NoBg=removeBackground(Q2Image2Path)
# Q2Image2Gender=Question2_DetectGender(Q2Image2Path, Q2Image2NoBg)
# print(f"the gender of image2 is {Q2Image2Gender}")

#Run 3-------------------------------------------------------
Q3ImagePath="fig4.jpg"

# image1=grayScaleImage(Q3ImagePath)
# image2=blurImage(Q3ImagePath)

# Question3_FindBlurAndGreyImage("blurImage.png","grayImg.png")
# Question3_FindBlurAndGreyImage(image1, image2)

#Run 4-------------------------------------------------------
Q4ImagePath="fig1.jpg"
# Question4_FindArea(Q4ImagePath)

#Run 5------------------------------------------------------
Q5ImagePath="redArrow.jpg"
# Question5_FindArrowPercent(Q5ImagePath)

#Run 6-------------------------------------------------------
colourTable = [
    {"color": (206, 70, 64), "outputPath": "segment1.jpg"},
    {"color": (114, 145, 200), "outputPath": "segment2.jpg"},
    {"color": (77, 177, 35), "outputPath": "segment3.jpg"},
    {"color": (209, 177, 136), "outputPath": "segment4.jpg"},
    {"color": (10, 202, 255), "outputPath": "segment5.jpg"}
    
]


for info in colourTable:
    color = info["color"]
    outputPath = info["outputPath"]
    tolerance = 20
    
    Question6_SegmentBones('finger-bones.jpg', color, tolerance, outputPath)






