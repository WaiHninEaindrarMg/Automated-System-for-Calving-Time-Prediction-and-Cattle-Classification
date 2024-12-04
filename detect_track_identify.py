import os, torch, cv2, joblib, glob, re, csv, time, json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas as read_csv
import ipywidgets as widgets
import matplotlib.pyplot as plt
import cv2 as cv
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import *
from tensorflow.keras.layers import Input,Flatten,Dense,Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from torchvision import transforms
from torch.utils.data import DataLoader

from CTA.CTA_iou_tracker import IOUTracker
from CTA.utils import draw_tracks_identify

from pandas.plotting import scatter_matrix
from scipy.ndimage.morphology import binary_erosion
from matplotlib import pyplot 
from PIL import Image, ImageDraw
from ultralytics import YOLO


cow_name = []
track_data = []
cow_data = []
track_id = []
track_bboxs = []
track_scores = []
track_counts={}
track_dict = {}
sec_frame_num = 0
track_id_to_labels = {}
labels_to_track_id = {}
image_count = 0 
fir_image_count = 0
fir_frame_count = 0
fir_frame_num = 0

calving_data = []
all_cow_mask_points = []
cropped_binary_mask_images = []
trk_bbox_list = []  

 
chosen_tracker = widgets.Select(
    options=["IOUTracker"],
    value='IOUTracker',
    rows=5,
    description='Customized Tracking Algorithm:',
    disabled=False
)

if chosen_tracker.value == 'IOUTracker':
    tracker = IOUTracker(max_lost=50000, iou_threshold=0.5, min_detection_confidence=0.8, max_detection_confidence=0.9, tracker_output_format='mot_challenge')
                        
else:
    print("Please choose one tracker from the above list.")

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


frameSkip = 15 # update the skip frame rate for processing

yolov8_config ="./model/detection_model/custom_data.yaml"
yolov8_weights ="./model/detection_model/best.pt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# video name
videos_path = './videos/1/*.mp4'


bbox_list = []
start_time = time.time()
cc = 0
ccc = 0

identify_model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
svm_model = joblib.load("./model/identification_model/svm_model.pkl")


# Define a function to extract feature
def extract_features(image):
    img = cv2.resize(image, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)  
    features = identify_model.predict(img)
    return features

for video_path_url in glob.glob(videos_path):

    pathList = video_path_url.split("\\")
    vdoName = pathList[len(pathList)-1].replace(".mp4","")
    print("video_name : ", vdoName)
    imageName = vdoName
    dirName = f'./{vdoName}'
    current_path = './videos/1/'

    cap = cv2.VideoCapture(video_path_url)
    # calculating total frame
    f_cnt = cv2.CAP_PROP_FRAME_COUNT
    numFrames = min(int(cap.get(f_cnt)), 450)
    print(numFrames)
    fir_count = 1
    fir_image_count = 1


    for i in range(0,numFrames,frameSkip):
        
        cap.set(cv2.CAP_PROP_POS_FRAMES,i)
        ret,frame = cap.read()
        
        if not ret:
            print("Can't read the frame")
            break
     
        img = frame
        img = cv2.resize(img, (1080, 1080))
        cv2.namedWindow("CTA_Test_First", cv2.WINDOW_NORMAL)

        # detection 
        detector = YOLO(yolov8_weights)
        outputs = detector.predict(img)

        bboxs = []
        scores = []
        class_ids = []
        masks = []

        for r in outputs:

            bboxs.extend(r.boxes.xyxy)
            scores.extend(r.boxes.conf)
            class_ids.extend(r.boxes.cls)
            masks.extend(r.masks.data)

        # # Compute the cumulative binary mask
        bboxs = np.array([bbox.cpu().numpy() for bbox in bboxs])
        scores = np.array([score.cpu().numpy() for score in scores])
        class_ids = np.array([class_id.cpu().numpy() for class_id in class_ids]) 


        masks_array = np.array([mask.cpu().numpy() for mask in masks])
        masks = np.array(masks_array)
        mask_sum = masks.sum(axis=0)
        mask_sum[mask_sum > 1] = 1
        mask_sum = (mask_sum*255).astype(np.uint8)


        replicated_mask = np.repeat(mask_sum[:, :, np.newaxis], 3, axis=2)
        replicated_mask_ = cv2.resize(replicated_mask, (1080, 1080))

        # Perform element-wise multiplication after broadcasting
        color_mask = img * replicated_mask_
    

        # Find contours from the mask_sum
        contours, _ = cv2.findContours(mask_sum, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert the RGB image (NumPy array) to a PIL Image
        pil_image = Image.fromarray(rgb_image)

        poly = Image.new('RGBA', pil_image.size)
        pdraw = ImageDraw.Draw(poly)

        # Find contours and draw polygons on the RGBA image
        for contour in contours:
             if len(contour) >= 3:  # Checking if there are enough points for a polygon
                polygon = [tuple(point[0]) for point in contour]
                pdraw.polygon(polygon, fill=(0, 0, 255, 128), outline=(0, 0, 0, 255))
                
        # Paste the drawn polygons onto the image after all contours are processed
        pil_image.paste(poly, mask=poly)


        # Calculate the area of each mask
        areas = np.array([mask.sum() for mask in masks])
        print("-------- Binary Mask Areas -------- : ", areas)

        # tracking
        tracks = tracker.update(bboxs,scores,class_ids)
        print("--------------------------------------")
        print(f"Tracking : ")

        label = []
        identify_label = []

       
        # Draw track ID labels with the corresponding cow colors and trajectories
        for trk in tracks:
            fir_frame_num = trk[0]
            trk_id = trk[1]+1              
            xmin = trk[2]
            ymin = trk[3]
            width = trk[4]
            height = trk[5]
            scores = trk[6]
            trk_bboxs = trk[2:6]

            print(f"BBOxs: {trk_bboxs}")

            # Assuming 'masks' is a list of binary mask images
            for idx, mask_img in enumerate(masks):
                cropped_binary_mask_image = mask_img * 255

                cropped_binary_mask_image_ = cropped_binary_mask_image[int(ymin):int(height), int(xmin):int(width)]
                cropped_binary_mask_image_resized = cv2.resize(cropped_binary_mask_image_, (color_mask.shape[1], color_mask.shape[0]))
                cropped_binary_mask_image = cropped_binary_mask_image.astype(np.uint8)
                cropped_binary_mask_image_replicated = np.repeat(cropped_binary_mask_image[:, :, np.newaxis], 3, axis=2)
                cropped_binary_mask_image_replicated_ = cv2.resize(cropped_binary_mask_image_replicated, (1080, 1080))
                cropped_color_mask_image = color_mask * cropped_binary_mask_image_replicated_
                white_pixel_count = np.count_nonzero(cropped_binary_mask_image_resized == 255)
                print(f"Track_ID: {trk_id}, White Pixels: {white_pixel_count}")

                if white_pixel_count > 2000:
                    cow_mask_points = np.transpose(np.where(cropped_binary_mask_image_resized == 255))
                    cropped_color_mask_image_ = color_mask[int(trk_bboxs[1]):int(trk_bboxs[3]), int(trk_bboxs[0]):int(trk_bboxs[2])]
                    
            # Identification
            features = extract_features(cropped_color_mask_image_)
            features_ = features.flatten()
            predictions = svm_model.predict([features_])  
            print(f"Type : {type(predictions[0])}")
            label.append(predictions[0])
            print("Predicted Label:", label)
            print(f"Type : {type(label[0])}")

            # Calculate the center coordinates of the bounding box
            cx = int((xmin + width) / 2)
            cy = int((ymin + height) / 2)
        
            track_data.append([trk_id, cx, cy, '{}_{}.jpg'.format(vdoName, fir_count)])
           
            track_id.append(trk_id)
            track_bboxs.append((xmin, ymin, width, height))
            track_scores.append(scores)

            if trk_id not in track_counts:

                track_counts[trk_id] = 1

            else:
                track_counts[trk_id] += 1


            if trk_id in track_dict:
                track_dict[trk_id].append((xmin, ymin, width, height))

            else:
                track_dict[trk_id] = [(xmin, ymin, width, height)]

        print(f"track_counts : {track_counts}")
        fir_frame_count = fir_frame_num


        for i in range(len(label)):
            print(f" I : {i}")
            current_track_id = track_id[i]
            print(f"current_track_id : {current_track_id}")
            current_label_id = label[i]
            print(f"current_label_id : {current_label_id}")


            # Check if the track_id is already in the dictionary
            if current_track_id in track_id_to_labels:
                track_id_to_labels[current_track_id].append(current_label_id)


            else:
                # If it's not in the dictionary, create a new list
                track_id_to_labels[current_track_id] = [current_label_id]


            # Check if the track_id is already in the dictionary
            if current_label_id in labels_to_track_id:
                labels_to_track_id[current_label_id].append(current_track_id)


            else:
                # If it's not in the dictionary, create a new list
                labels_to_track_id[current_label_id] = [current_track_id]
                

        # Print the final result
        print("#################################################")
        print(f"Labels_to_Track_ID : {labels_to_track_id}")
        print(f"Track_ID_to_Labels : {track_id_to_labels}")

        print("#################################################")


        # Initialize a dictionary to store the number with the maximum count for each label
        max_number_per_label_first = {}

        # Iterate through the labels_to_track_id dictionary
        for label_id, track_ids in labels_to_track_id.items():
            number_counts = {}  # Dictionary to store the count of each number
            for number in track_ids:
                if number in number_counts:
                    number_counts[number] += 1
                else:
                    number_counts[number] = 1
            max_number = max(number_counts, key=number_counts.get)  # Find the number with the maximum count
            max_number_per_label_first[label_id] = max_number

        print(f"Maximum Count ID per Global-ID: {max_number_per_label_first}")

        # Initialize a dictionary to store the number with the maximum count for each label
        max_number_per_trackID_first = {}

        # Iterate through the labels_to_track_id dictionary
        for trkk_id, label_ids in track_id_to_labels.items():
            number_counts_ = {}  # Dictionary to store the count of each number
            for number_ in label_ids:
                if number_ in number_counts_:
                    number_counts_[number_] += 1
                else:
                    number_counts_[number_] = 1
            max_number_ = max(number_counts_, key=number_counts_.get)  # Find the number with the maximum count
            max_number_per_trackID_first[trkk_id] = max_number_

        print(f"Maximum Count ID per Track-ID: {max_number_per_trackID_first}")

        print(f"Image Count : {fir_image_count}")

        if fir_image_count > 30:
            break

        else:
            print(f"Maximum Count ID per Track-ID: {max_number_per_trackID_first}")
            print(f"Maximum Count ID per Global-ID: {max_number_per_label_first}")

        display_img_ = draw_tracks_identify(img,pil_image,tracks,label)
        # display_img_ = draw_tracks(img,tracks)

        # # Show images
        cv2.imshow("CTA_Test_First", display_img_)
        fir_image_count = fir_image_count+1


        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

cv2.destroyAllWindows()

print("First 30 Frames Finished !!!")
print("###################################################################")
print("###################################################################")
print("###################################################################")

for video_path_url in glob.glob(videos_path):

    pathList = video_path_url.split("\\")
    vdoName = pathList[len(pathList)-1].replace(".mp4","")
    print("video_name : ", vdoName)
    imageName = vdoName
    dirName = f'./{vdoName}'
    current_path = './videos/1/'
    choose_write_folder = os.path.join(current_path, "tracking", dirName)  


    csvName = pathList[len(pathList)-1].replace(".mp4","")
    print("csv_name : ", csvName)
       
    if not os.path.exists(choose_write_folder):
        print("Directory", choose_write_folder, "Created")
        os.makedirs(choose_write_folder)
    else:
        print("Directory", choose_write_folder, "already exists")


    print(f"video directory path {video_path_url}")    
    resultVdoName = choose_write_folder + f'/{vdoName}.mp4'
    out = cv2.VideoWriter(resultVdoName, cv2.VideoWriter_fourcc(*'mp4v'), 5, (1080, 1080))          

    resultCsvName = choose_write_folder + f'/{csvName}.csv'
    plotCsvName = choose_write_folder + f'/{csvName}_plot.csv'
    calvingCsvName = choose_write_folder + f'/{csvName}_calving.csv'
    print(f"calvingCsvName : {calvingCsvName}")

    cap = cv2.VideoCapture(video_path_url)
    f_cnt = cv2.CAP_PROP_FRAME_COUNT
    numFrames = int(cap.get(f_cnt))
    print(numFrames)
    sec_count = 1

    for i in range(0,numFrames,frameSkip):
        
        cap.set(cv2.CAP_PROP_POS_FRAMES,i)
        ret,frame = cap.read()
        
        if not ret:
            print("Can't read the frame")
            break
     
        img = frame
        img = cv2.resize(img, (1080, 1080))

        cv2.namedWindow("CTA_Test", cv2.WINDOW_NORMAL)

        # detection 
        detector = YOLO(yolov8_weights)
        outputs = detector.predict(img)

        bboxs = []
        scores = []
        class_ids = []
        masks = []

        for r in outputs:

            bboxs.extend(r.boxes.xyxy)
            scores.extend(r.boxes.conf)
            class_ids.extend(r.boxes.cls)
            masks.extend(r.masks.data)

        # Compute the cumulative binary mask

        bboxs = np.array([bbox.cpu().numpy() for bbox in bboxs])
        scores = np.array([score.cpu().numpy() for score in scores])
        class_ids = np.array([class_id.cpu().numpy() for class_id in class_ids]) 


        masks_array = np.array([mask.cpu().numpy() for mask in masks])
        masks = np.array(masks_array)
        mask_sum = masks.sum(axis=0)
        mask_sum[mask_sum > 1] = 1
        mask_sum = (mask_sum*255).astype(np.uint8)

        replicated_mask = np.repeat(mask_sum[:, :, np.newaxis], 3, axis=2)

        replicated_mask_ = cv2.resize(replicated_mask, (1080, 1080))

        # Perform element-wise multiplication after broadcasting
        color_mask = img * replicated_mask_

        # Find contours from the mask_sum
        contours, _ = cv2.findContours(mask_sum, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert the RGB image (NumPy array) to a PIL Image
        pil_image = Image.fromarray(rgb_image)

        poly = Image.new('RGBA', pil_image.size)
        pdraw = ImageDraw.Draw(poly)

        # Find contours and draw polygons on the RGBA image
        for contour in contours:
            if len(contour) >= 3:  # Checking if there are enough points for a polygon
                polygon = [tuple(point[0]) for point in contour]
                pdraw.polygon(polygon, fill=(0, 0, 255, 128), outline=(0, 0, 0, 255))

        # Paste the drawn polygons onto the image after all contours are processed
        pil_image.paste(poly, mask=poly)

        # Calculate the area of each mask
        areas = np.array([mask.sum() for mask in masks])
        print("-------- Binary Mask Areas -------- : ", areas)

        # tracking
        tracks = tracker.update(bboxs,scores,class_ids)
        print("--------------------------------------")
        print(f"Tracking : ")

        label = []
        identify_label = []

        for t in tracks:
            print(t)
        print("--------------------------------------")

       
        # Draw track ID labels with the corresponding cow colors and trajectories
        for trk in tracks:
            fir_frame_num = trk[0]
            trk_id = trk[1]+1              
            xmin = trk[2]
            ymin = trk[3]
            width = trk[4]
            height = trk[5]
            scores = trk[6]
            trk_bboxs = trk[2:6]

            print(f"BBOxs: {trk_bboxs}")

            # Assuming 'masks' is a list of binary mask images
            for idx, mask_img in enumerate(masks):
                cropped_binary_mask_image = mask_img * 255

                cropped_binary_mask_image_ = cropped_binary_mask_image[int(ymin):int(height), int(xmin):int(width)]


                # Resize the binary mask to match the shape of the color mask
                cropped_binary_mask_image_resized = cv2.resize(cropped_binary_mask_image_, (color_mask.shape[1], color_mask.shape[0]))

                # Replicate the single-channel mask into a 3-channel mask
                cropped_binary_mask_image = cropped_binary_mask_image.astype(np.uint8)
                cropped_binary_mask_image_replicated = np.repeat(cropped_binary_mask_image[:, :, np.newaxis], 3, axis=2)

                cropped_binary_mask_image_replicated_ = cv2.resize(cropped_binary_mask_image_replicated, (1080, 1080))

                # Perform the element-wise multiplication
                cropped_color_mask_image = color_mask * cropped_binary_mask_image_replicated_

                # Count white pixels (pixel value 1) in the binary mask image
                white_pixel_count = np.count_nonzero(cropped_binary_mask_image_resized == 255)
                print(f"Track_ID: {trk_id}, White Pixels: {white_pixel_count}")

                if white_pixel_count > 2000:
                    cow_mask_points = np.transpose(np.where(cropped_binary_mask_image_resized == 255))
                    # Save the cropped image to the respective track ID folder
                    track_id_path = f'{trk_id}'
                    binary_track_folder_path = os.path.join(choose_write_folder, "Binary", track_id_path)

                    if not os.path.exists(binary_track_folder_path):
                        print("Directory", binary_track_folder_path, "Created")
                        os.makedirs(binary_track_folder_path)
                    else:
                        print("Directory", binary_track_folder_path, "already exists")

                    print(f"Second Image Count : {sec_count}")

                    image_filename = f'{vdoName}_{sec_count}.jpg'
                    image_filepath = os.path.join(binary_track_folder_path, image_filename)

                    # Save the binary mask image
                    cv2.imwrite(image_filepath, cropped_binary_mask_image)


                    print(f" Cow Mask Points : {cow_mask_points} ")
                    calving_data.append([trk_id, (xmin, ymin, width, height), cow_mask_points, '{}_{}.jpg'.format(vdoName, sec_count)])


                    cropped_color_mask_image_ = cropped_color_mask_image[int(trk_bboxs[1]):int(trk_bboxs[3]), int(trk_bboxs[0]):int(trk_bboxs[2])]
                
                    color_binary_track_folder_path = os.path.join(choose_write_folder, "Color-Crop", track_id_path)
                    print(f"track_folder_path : {color_binary_track_folder_path}")

                    if not os.path.exists(color_binary_track_folder_path):
                        print("Directory", color_binary_track_folder_path, "Created")
                        os.makedirs(color_binary_track_folder_path)
                    else:
                        print("Directory", color_binary_track_folder_path, "already exists")

                    image_filename = f'{vdoName}_{sec_count}.jpg'
                    image_filepath = os.path.join(color_binary_track_folder_path, image_filename)
                    cv2.imwrite(image_filepath, cropped_color_mask_image_)

                    color_binary_track_folder_path_ = os.path.join(choose_write_folder, "Color", track_id_path)
                    print(f"track_folder_path : {color_binary_track_folder_path_}")

                    if not os.path.exists(color_binary_track_folder_path_):
                        print("Directory", color_binary_track_folder_path_, "Created")
                        os.makedirs(color_binary_track_folder_path_)
                    else:
                        print("Directory", color_binary_track_folder_path_, "already exists")

                    image_filename_ = f'{vdoName}_{sec_count}.jpg'
                    image_filepath_ = os.path.join(color_binary_track_folder_path_, image_filename_)
                    cv2.imwrite(image_filepath_, cropped_color_mask_image)


            # Calculate the center coordinates of the bounding box
            cx = int((xmin + width) / 2)
            cy = int((ymin + height) / 2)
        
            track_data.append([trk_id, cx, cy, '{}_{}.jpg'.format(vdoName, sec_count)])
           
            track_id.append(trk_id)
            track_bboxs.append((xmin, ymin, width, height))
            track_scores.append(scores)

            if trk_id not in track_counts:

                track_counts[trk_id] = 1

            else:
                track_counts[trk_id] += 1


            if trk_id in track_dict:
                track_dict[trk_id].append((xmin, ymin, width, height))

            else:
                track_dict[trk_id] = [(xmin, ymin, width, height)]

        print(f"track_counts : {track_counts}")
        sec_frame_count = sec_frame_num


        if sec_frame_count % 30 == 0:
            track_dict = {}
            
        else:
            pass
            print(f"Track dict : {track_dict}")


        # Get track IDs with count smaller than frame count
        tracks_to_check = [track_id for track_id, count in track_counts.items() if sec_count < sec_frame_count]

        print(f"Second Program Maximum Count ID per Track-ID: {max_number_per_trackID_first}")
        print(f"Second Program Maximum Count ID per Global-ID: {max_number_per_label_first}")

        for labelID, trID in max_number_per_label_first.items(): 
            print(f"Current Track ID : {current_track_id}, Track_ID: {trID}, Label_ID: {labelID}")

            if current_track_id > 0:
                identify_label.append(labelID)
                print(f"Current Track : {current_track_id}, identify_label : {identify_label}")
            else:
                print("Else ------------------------------------------------------------------")
        print(f"Exist For loop!")
        print(f"identify_label : {identify_label}")

        # Write the data to the CSV file
        with open(plotCsvName, "a", newline="") as file:
            writer = csv.writer(file)
            if cc == 0:
                writer.writerow(["Track ID", "Gravity X_Values", "Gravity Y_Values", "Frame"])  # Write the header row
            writer.writerows(track_data)  
            track_data = []

        display_img = draw_tracks_identify(img,pil_image,tracks,identify_label)

        numpy_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)  # Convert PIL to NumPy
 
        save_video_path = (os.path.join(choose_write_folder, '{}_{}.jpg'.format(vdoName, sec_count)))

        if sec_count != 0:
            cow_name.append('{}_{}.jpg'.format(vdoName, sec_count))
        
        with open(resultCsvName, "a", newline="") as file1:
            writer1 = csv.writer(file1)
            if cc == 0:
                writer1.writerow(["ID", "Bboxes", "Scores","Image","GT"])  
                cc +=1
            writer1.writerow([track_id, track_bboxs, track_scores,cow_name,len(track_bboxs)])

            track_id = []
            track_bboxs = []
            track_scores = []
            cow_name = []

        with open(calvingCsvName, "a", newline="") as file2:
            writer2 = csv.writer(file2)
            if ccc == 0:
                writer2.writerow(["Track-ID", "Bboxes", "Masks","Image-Name"])  
                ccc +=1
            writer2.writerows(calving_data)  # Write the track data
            calving_data = []
            all_cow_mask_points = []

        label = f" Frame Num : {sec_count}"
        cv2.putText(display_img, label, (61,78), cv2.FONT_HERSHEY_SIMPLEX, 2.5, [255, 255, 255], 5)
        cv2.imwrite(save_video_path, display_img)
        out.write(display_img)

        # Show images
        cv2.imshow("CTA_Test", display_img)


        sec_count = sec_count+1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


end_time = time.time()

test_duration = end_time - start_time

hours, remainder = divmod(test_duration, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Testing Duration : {int(hours):02d} : {int(minutes):02d} : {int(seconds):02d}")
cv2.destroyAllWindows()






