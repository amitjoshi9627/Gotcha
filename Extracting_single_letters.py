import os 
import os.path, glob, imutils
import cv2

def extract_single_letter(captcha_image_folder = "generated_captcha_images",output_folder = "extracted_letter_images"):
    
    # Getting a list of all the captcha images we need to process
    captcha_image_files = glob.glob(os.path.join(captcha_image_folder,'*'))

    counts = {}

    # loop over the image paths
    for i,image_file in enumerate(captcha_image_files):
        #print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))
        
        # grab the base filename as the text
        file_name = os.path.basename(image_file)
        correct_text = os.path.splitext(file_name)[0]
        
        # Load the image
        image = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
        
        # Addding some extra padding around the image
        image = cv2.copyMakeBorder(image,8,8,8,8,cv2.BORDER_REPLICATE)
        
        # threshold the image (convert it to pure black and white)
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        #Finding the contours
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contours = contours[0] if imutils.is_cv2() else contours[1]
        letter_image_regions = []
        
        # Now we can loop through each of the four contours and extract the letter inside of each one
        for contour in contours:
            
            (x,y,w,h) = cv2.boundingRect(contour)
            if h == 0:
                print(x,y,w,h)
            if w/h > 1.25:
                
                # This contour is too wide to be a single letter!
                # Split it in half into two letter regions!
                
                half_width = int(w/2)
                letter_image_regions.append((x, y, half_width, h))
                letter_image_regions.append((x + half_width, y, half_width, h))
                
            else:
                letter_image_regions.append((x,y,w,h))
                
        if len(letter_image_regions)>4:
            continue
        
        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
        
        for box,text in zip(letter_image_regions, correct_text):
            
            (x,y,w,h) = box
            
            letter_image = image[y-2: y+h+2,x-2:x+w+2]
            saving_path = os.path.join(output_folder,text)
            
            # If the folder doesn't exists make it
            if not os.path.exists(saving_path):
                os.makedirs(saving_path)
            count = counts.get(text,1)
            
            # Writing image to file
            p = os.path.join(saving_path,f'{str(count).zfill(6)}.png')
            cv2.imwrite(p,letter_image)        
            counts[text]= count + 1
