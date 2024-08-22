import cv2

# Load the ideal image
ideal_img = cv2.imread('samples/ideal.jpg', cv2.IMREAD_GRAYSCALE)

# List of images
sample_images = ['samples/sample2.jpg', 'samples/sample3.jpg', 'samples/sample4.jpg', 'samples/sample5.jpg', 'samples/sample6.jpg']

# Function to find the contours and defects
def find_defects(ideal, sample):
    # Thresholding that create binary
    _, ideal_thresh = cv2.threshold(ideal, 127, 255, cv2.THRESH_BINARY)
    _, sample_thresh = cv2.threshold(sample, 127, 255, cv2.THRESH_BINARY)
    
    diff = cv2.bitwise_xor(ideal_thresh, sample_thresh)
    contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_defects = len(contours)
    
    ideal_contours, _ = cv2.findContours(ideal_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sample_contours, _ = cv2.findContours(sample_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ideal_inner_circle = cv2.minEnclosingCircle(max(ideal_contours, key=cv2.contourArea))
    sample_inner_circle = cv2.minEnclosingCircle(max(sample_contours, key=cv2.contourArea))
    
    ideal_diameter = ideal_inner_circle[1] * 2
    sample_diameter = sample_inner_circle[1] * 2
    
    if sample_diameter > ideal_diameter:
        diameter_status = "larger"
    elif sample_diameter < ideal_diameter:
        diameter_status = "smaller"
    else:
        diameter_status = "identical"
    
    return num_defects, diameter_status

# for loop each sample image
for sample_image in sample_images:
    sample_img = cv2.imread(sample_image, cv2.IMREAD_GRAYSCALE)

    defects, diameter_status = find_defects(ideal_img, sample_img)
    
    print(f"Results for {sample_image}:")
    print(f"Number of broken or worn teeth: {defects}")
    print(f"Inner diameter is {diameter_status} compared to the ideal gear.")
    print("-" * 40)
