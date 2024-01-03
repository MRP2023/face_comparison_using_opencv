from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import os

# Create your views here.
from django.http import JsonResponse
# from django.utils import compare_images

import cv2
import dlib
import numpy as np


@csrf_exempt                                                                                                                                                                                                                                                                                                                        
def compare_images_api(request):
    if request.method == 'POST':
        
        raw_data = request.body

        # # Decode the raw data based on content type (assuming it's JSON)
        # try:
        #     decoded_data = raw_data.decode('utf-8')
        #     # Now, 'decoded_data' contains the JSON or other data sent in the request body
        #     print(decoded_data)
        # except UnicodeDecodeError:
        #     return JsonResponse({'error': 'Unable to decode raw data'})

        # # Rest of your code for image comparison
        # print(raw_data)

    
        # decoded_data = raw_data.decode('utf-8')
        # print(decoded_data)
        
        
        
        image1 = request.FILES.get('image1')
        image2 = request.FILES.get('image2')

        if image1 and image2:
            image1_path = default_storage.save(os.path.join('uploads', image1.name), image1)
            image2_path = default_storage.save(os.path.join('uploads', image2.name), image2)
        else:
            return JsonResponse({'error': 'Image1 or Image2 not provided'})

# Rest of your code for image comparison


        result = compare_images(image1_path, image2_path)

        
        default_storage.delete(image1_path)
        default_storage.delete(image2_path)

        return JsonResponse(result)

    return JsonResponse({'error': 'Invalid request method'})

def handle_uploaded_file(f):
    row_file = None
    for chunk in f.chunks():
        row_file.write(chunk)

def compare_images(image1_path, image2_path):
    try:
        print(image1_path, image2_path)
        # Read the images
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)

        # Convert images to grayscale
        gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Initialize face detector and shape predictor from dlib
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("trained_model/shape_predictor_68_face_landmarks.dat")

        # Detect faces in the images
        faces1 = detector(gray_image1)
        faces2 = detector(gray_image2)

        if len(faces1) > 0 and len(faces2) > 0:
            # Get facial landmarks for the first face
            shape1 = predictor(gray_image1, faces1[0])
            landmarks1 = np.array([[shape1.part(i).x, shape1.part(i).y] for i in range(68)])

            # Get facial landmarks for the second face
            shape2 = predictor(gray_image2, faces2[0])
            landmarks2 = np.array([[shape2.part(i).x, shape2.part(i).y] for i in range(68)])

            # Manually add 59 additional landmarks to reach 127
            additional_landmarks = np.array([[0, 0] for _ in range(59)])

            # Combine the original and additional landmarks
            landmarks1 = np.concatenate((landmarks1, additional_landmarks))
            landmarks2 = np.concatenate((landmarks2, additional_landmarks))

            # Calculate the Euclidean distance between corresponding landmarks
            distances = np.linalg.norm(landmarks1 - landmarks2, axis=1)

            # Calculate the similarity percentage based on the average distance
            similarity_percentage = 100 - (np.average(distances) / 10)

            print({'success': True, 'similarity_percentage': similarity_percentage})
            
            return {'success': True, 'similarity_percentage': similarity_percentage}
            
            
            # # Draw landmarks on the images
            # image1_landmarks = image1.copy()
            # image2_landmarks = image2.copy()

            # for x, y in landmarks1.astype(int):
            #     cv2.circle(image1_landmarks, (x, y), marker_size, (0, 255, 0), -1)

            # for x, y in landmarks2.astype(int):
            #     cv2.circle(image2_landmarks, (x, y), marker_size, (0, 255, 0), -1)
            
            # # Resize the images with landmarks
            # resized_image1 = resize_image(image1_landmarks, scale_percent)
            # resized_image2 = resize_image(image2_landmarks, scale_percent)
            
            
            
            # # Display the images with landmarks
            # # cv2.imshow("Image 1 with Landmarks", image1_landmarks)
            # # cv2.imshow("Image 2 with Landmarks", image2_landmarks)
            # cv2.imshow("Image 1 with Landmarks", resized_image1)
            # cv2.imshow("Image 2 with Landmarks", resized_image2)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            
            
        else:
            print({'success': False, 'error': 'No faces detected in one or both images'})
            return {'success': False, 'error': 'No faces detected in one or both images'}
    except Exception as e:
        print({'success': False, 'error': str(e)})
        return {'success': False, 'error': str(e)}
    