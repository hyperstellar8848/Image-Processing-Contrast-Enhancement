import cv2
import matplotlib.pyplot as plt

image_name = 'first_image.jpg'    

# Load the image
img = cv2.imread(image_name)

if img is None:
    print("Error: Could not load image. Check the filename!")
else:
    # Convert to Grayscale (like rgb2gray in matlab)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Adjust contrast (like imadjust in matlab)
    adjusted = cv2.equalizeHist(gray)
    
    # Show results
    plt.figure(figsize=(15, 8))
    plt.suptitle(f'Processing: {image_name}', fontsize=14)
    
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original RGB')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(adjusted, cmap='gray')
    plt.title('After Contrast Adjust')
    plt.axis('off')
    
    # Histograms
    plt.subplot(2, 3, 4)
    plt.hist(img.ravel(), 256, [0, 256])
    plt.title('Original Histogram')
    
    plt.subplot(2, 3, 5)
    plt.hist(gray.ravel(), 256, [0, 256])
    plt.title('Grayscale Histogram')
    
    plt.subplot(2, 3, 6)
    plt.hist(adjusted.ravel(), 256, [0, 256])
    plt.title('Adjusted Histogram')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Done processing: {image_name}")
