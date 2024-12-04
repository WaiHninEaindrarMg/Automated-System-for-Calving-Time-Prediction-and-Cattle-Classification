from CTA.utils.misc import iou_xywh as iou
from CTA.tracker import Tracker
from CTA.track import Track

class IOUTracker(Tracker):
   
    def __init__(
            self,
            max_lost=2,
            iou_threshold=0.5,
            min_detection_confidence=0.4,
            max_detection_confidence=0.7,
            tracker_output_format='CTA_challenge'

    ):
        self.iou_threshold = iou_threshold
        self.max_detection_confidence = max_detection_confidence
        self.min_detection_confidence = min_detection_confidence
        self.track_dict = {}
        self.track_counts = {}
        self.num = 0
        self.last_values = []
        self.miss_index = []
        self.missing_bboxes = {}
        self.missing_numbers = []
        self.occlusion_ids = {}
        self.current_track_id = [[0,1,2,3],]
        self.new_ids = []
       

        super(IOUTracker, self).__init__(max_lost=max_lost, tracker_output_format=tracker_output_format)

    def calculate_iou(self, bbox1, bbox2):
        # Calculate the coordinates of the intersection rectangle
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        # If there is no intersection, return 0
        if x2 <= x1 or y2 <= y1:
            return 0.0

        # Calculate the area of intersection rectangle
        intersection_area = (x2 - x1) * (y2 - y1)

        # Calculate the area of both bounding boxes
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        # Calculate the area of union
        union_area = bbox1_area + bbox2_area - intersection_area

        # Calculate the IoU and return the result
        iouu = intersection_area / union_area
        return iouu

    # 4 cow-ids
    def update(self, bboxes, detection_scores, class_ids, **kwargs):

        detections = Tracker.preprocess_input(bboxes, class_ids, detection_scores)
        detections_ = Tracker.preprocess_input(bboxes, class_ids, detection_scores)

        num_cows = 4

        self.num = len(detections)
        # print(f"len of detection : {self.num}")

        self.frame_count += 1
        track_ids = list(self.tracks.keys())

        updated_tracks = []

        for track_id in track_ids:
            
            if len(detections) > 0:
                #print(f"len(detections) > 0:")
                idx, best_match = max(enumerate(detections), key=lambda x: iou(self.tracks[track_id].bbox, x[1][0]))
                (bb, cid, scr) = best_match

                if iou(self.tracks[track_id].bbox, bb) > self.iou_threshold:
                    self._update_track(track_id, self.frame_count, bb, scr, class_id=cid,
                                       iou_score=iou(self.tracks[track_id].bbox, bb))
                    updated_tracks.append(track_id)
                    del detections[idx]


            if len(updated_tracks) == 0 or track_id is not updated_tracks[-1]:
                self.tracks[track_id].lost += 1
                if self.tracks[track_id].lost > self.max_lost:
                    self._remove_track(track_id)
        
        for bb, cid, scr in detections:
            print("------------------------------------------------------------------------------")
            print("ADD Track")
            print("------------------------------------------------------------------------------")
            self._add_track(self.frame_count, bb, scr, self.num, self.track_dict, self.track_counts, self.occlusion_ids, class_id=cid)
            
        outputs = self._get_tracks(self.tracks)

        print(f"len of detection : {len(detections_)}")
        detected_num = len(detections_)


        # Find Occlusion Case
        if detected_num < num_cows:                          
            print("Found Occlusion")
            keys_in_order = list(self.track_dict.keys())
            print(f"List of Track Dict : {keys_in_order}")

            for track_id in keys_in_order:
                # Check if the track ID exists in the dictionary
                if track_id in self.track_dict:
                    # Get the last value for the current track ID
                    last_value = self.track_dict[track_id][-1]
                    self.last_values.append(last_value)

            # Print the list of last values
            print("-----------------------------------------------------")
            print("List of Last Values:")
            print(self.last_values)
            print("-----------------------------------------------------")


            current_ids = [track[1] for track in outputs]
            print(f"Current Track IDs : {current_ids}") 
            current_trk = [track[1:6] for track in outputs]
            print(f"Current Track : {current_trk}") 
            self.current_track_id.append(current_ids)

            # # Convert the lists to sets
            prev_track_ids = set(self.current_track_id[-2])
            print(f"prev_track_ids : {prev_track_ids}")
            current_track_ids = set(self.current_track_id[-1])
            print(f"current_track_ids : {current_track_ids}")


            excluded_numbers = current_ids
            print(f"excluded numbers : {excluded_numbers}")        
                
            for num in range(0,num_cows):                                
                if num not in excluded_numbers:
                    #print("Miss ID :",num)
                    self.missing_numbers.append(num)
            print("Miss ID List :",self.missing_numbers)

            # Find out new_ids.
            new_numbers = [index for index, item in enumerate(self.current_track_id[-1]) if item not in self.current_track_id[-2]]

            for index in new_numbers:
                ind = self.current_track_id[-1][index]
                print(f"New ID : {ind}")

                try:
                    ind_bbox = current_trk[ind]
                    print(f"New ID BBOx : {ind_bbox[1:]}")

                except IndexError:
                    ind_bbox = None  # or any other value you want to use as a default

                threshold = 0.8
                for id_, bbox in self.occlusion_ids.items():
                    #print(f"track_bbox : {track_bbox}, track_bbox[1:] : {track_bbox[1:]}")

                    if ind_bbox is not None:
                        iou_result = self.calculate_iou(ind_bbox[1:], bbox)


                    else:
                        iou_result = 0.0  # or any other default value

                    print(f"IOU with New ID {ind} with Occlusion ID {id_}, {bbox}:", iou_result)

                    if iou_result > threshold:
                        print(f"iou_result : {iou_result} > threshold")

                        self._add_track_1(self.frame_count, ind_bbox[1:], 0.8, ind, id_, class_id=cid)

                print("---------------------------------------------------")

            for i in self.missing_numbers:
                try:
                    index = keys_in_order.index(i)
                    self.miss_index.append(index)
                except ValueError:
                    pass

            print(f"Missing Index: {self.miss_index}")

            for i in self.miss_index:
                print(f"{i} BBox: {self.last_values[i]}")
                self.missing_bboxes[i] = self.last_values[i]

            print(f"BBox: {self.missing_bboxes}")

            threshold = 0.05
        
            # Calculate and print the IoU for each missing bounding box with the current track bounding boxes
            for missing_index in self.missing_bboxes:
                # If missing_ids lies in Occlusion_ids, the program will break.
                if missing_index in self.occlusion_ids:
                    print(f"Missing index : {missing_index} lies in Occluded Numbers : {self.occlusion_ids}")
                    break
                missing_bbox = self.missing_bboxes[missing_index]
                print(f"Missing BBox {missing_index}:", missing_bbox)
                for track_bbox in current_trk:
                    #print(f"track_bbox : {track_bbox}, track_bbox[1:] : {track_bbox[1:]}")
                    iou_result = self.calculate_iou(missing_bbox, track_bbox[1:])
                    print(f"IOU with {missing_index} Track {track_bbox[0]}:", iou_result)

                    if iou_result > threshold:
                        self.occlusion_ids[missing_index] = missing_bbox   #save_occludee
                        self.occlusion_ids[track_bbox[0]] = track_bbox[1:] #save_occluder
                        print(f"Occlusion IDs BBoxs :", self.occlusion_ids)
                        print(f"Occluder : {track_bbox[0]} and Occludee : {missing_index}")
                print("---------------------------------------------------")


            self.missing_numbers = []
            self.missing_bboxes = {} 
            self.last_values = []
            self.miss_index = []
            


        else:
            self.current_track_id = [[0]]                   
 
        for trk in outputs:
            frame_num = trk[0]
            trk_id = trk[1]  #+1
            xmin = trk[2]
            ymin = trk[3]
            width = trk[4]
            height = trk[5]

            if trk_id in self.track_dict:
                self.track_dict[trk_id].append((xmin, ymin, width, height))

            else:
                self.track_dict[trk_id] = [(xmin, ymin, width, height)]

            if trk_id not in self.track_counts:
                self.track_counts[trk_id] = 1

            else:
                self.track_counts[trk_id] += 1

        print("######################################") 
        print(f"Track Counts : {self.track_counts}") 
        
        print(self.track_dict)

        if self.frame_count % 300 == 0:
            self.track_dict = {}

        else:
            pass

        return outputs
    
    
    def draw(image, detections):
        image_copy = image.copy()
        for box in detections:
            class_name = box['class']
            conf = box['confidence']
            text = ''
            if 'text' in box:
                text = box['text']
                if len(text) > 50:
                    text = text[:50] + ' ...'
            label = (str(box['id']) + '. ' if 'id' in box else '') + class_name + ' ' + str(int(conf*100)) + '%' + ((' | ' + text) if ('text' in box and len(box['text']) > 0 and not box['text'].isspace()) else '')
            width = box['width']
            height = box['height']
            color = box['color']

            if isinstance(color, str):
                color = ImageColor.getrgb(color)
                color = (color[2], color[1], color[0])
            
            top_left_point = {'x':box['x'], 'y':box['y']}
            bottom_right_point = {'x':box['x'] + width, 'y':box['y'] + height}
            image_copy = plot_box(image_copy, top_left_point, bottom_right_point, width, height, label, color=color)
        return image_copy
