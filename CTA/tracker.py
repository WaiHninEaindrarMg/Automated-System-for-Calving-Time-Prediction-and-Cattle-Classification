from collections import OrderedDict
import numpy as np
from scipy.spatial import distance
from CTA.utils.misc import get_centroid
from CTA.track import Track


class Tracker:

    def __init__(self, max_lost=5, tracker_output_format='CTA_challenge'):
        self.next_track_id = 0
        self.is_first_frame = True
        self.tracks = OrderedDict()
        self.max_lost = max_lost
        self.frame_count = 0
        self.tracker_output_format = tracker_output_format


    #Re_ID code in add_track_1
    def _add_track_1(self, frame_id, bbox, detection_confidence, old_id, new_id, class_id, **kwargs):

        del self.tracks[old_id]
        print(f"Deleted track with ID: {old_id}")
                
        self.next_track_id = new_id
        print(f"Reassign track with Occluded ID: {self.next_track_id}")

        self.tracks[self.next_track_id] = Track(
            self.next_track_id, frame_id, bbox, detection_confidence, class_id=class_id,
            data_output_format=self.tracker_output_format,
            **kwargs
        )
        print(f"Tracks list in Add Track 1 : {self._get_tracks(self.tracks)}")


    #Re_ID code in add_track
    def _add_track(self, frame_id, bbox, detection_confidence, detected_num, track_dict, track_counts, occlusion_ids, class_id, **kwargs):

        self.tracks[self.next_track_id] = Track(
            self.next_track_id, frame_id, bbox, detection_confidence, class_id=class_id,
            data_output_format=self.tracker_output_format,
            **kwargs
        )
        BBox_New = bbox
        print(f"New BBoX : {BBox_New}")

        print(f"Detected_Num : {detected_num}")

        num_cows = 4
        less_num_cows = 3

        Tracks = self._get_tracks(self.tracks)

        first_values = [track[0] for track in Tracks]
        print(f"First Frames :{first_values}")

        track_ids = [track[1] for track in Tracks]
        print(f"Track IDs : {track_ids}")

        last_track_id = track_ids[-1]
        print(f"Last Track ID : {last_track_id}")

        excluded_numbers = track_ids[0:-1]
        print(f"excluded numbers : {excluded_numbers}")  

        excluded_numbers_ = track_ids[0::]
        print(f"excluded numbers_ : {excluded_numbers_}")   

        if first_values[0] < num_cows:          
            self.next_track_id += 1
            print(f"Increment ID added : {self.next_track_id}")
        else:
            print(f"this is greater than 2")
            print(f"Next ID : {self.next_track_id}")
        

        # Re-ID assignment for ID increment 
        # Find missing ids, delete increment id and replace ids
        if last_track_id > less_num_cows:    #1
            
            print(f"excluded_numbers : {excluded_numbers}")
            for num in range(0,num_cows):    #2
                if num not in excluded_numbers:
                    print("Found ID :",num)
                    del self.tracks[self.next_track_id]
                    print(f"Deleted track with ID: {self.next_track_id}")
                    
                    print(f"ID reassign : {num}")
                    self.next_track_id = num
                    self.tracks[self.next_track_id] = Track(
                        self.next_track_id, frame_id, BBox_New, detection_confidence, class_id=class_id,
                        data_output_format=self.tracker_output_format,
                        **kwargs
                    )
                    print(f"Tracks list First : {self._get_tracks(self.tracks)}")
                    break

        # Find delete increment id
        if detected_num > num_cows and last_track_id > less_num_cows:                             
            del self.tracks[last_track_id]
            print(f"Deleted track with ID: {last_track_id}")
        
        # Find missing ids, and replace ids
        if first_values[0] > num_cows and len(track_ids) < detected_num:           
            threshold = 0.8
            for id_, bbox in self.occlusion_ids.items():
                iou_result = self.calculate_iou(BBox_New, bbox)
                print(f"IOU with New BBOx with Occlusion ID {id_}, {bbox}:", iou_result)

                if iou_result > threshold:
                    print(f"iou_result : {iou_result} > threshold")

            print("Second Condition")
            for num in range(0,num_cows):                         
                if num not in excluded_numbers_:
                    print("Found ID :",num)
                    print(f"ID reassign : {num}")
                    self.next_track_id = num
                    self.tracks[self.next_track_id] = Track(
                        self.next_track_id, frame_id, BBox_New, detection_confidence, class_id=class_id,
                        data_output_format=self.tracker_output_format,
                        **kwargs
                    )
                    print(f"Tracks list Second : {self._get_tracks(self.tracks)}")
                    break


    def _remove_track(self, track_id):
        
        del self.tracks[track_id]

    def _update_track(self, track_id, frame_id, bbox, detection_confidence, class_id, lost=0, iou_score=0., **kwargs):
        
        self.tracks[track_id].update(
            frame_id, bbox, detection_confidence, class_id=class_id, lost=lost, iou_score=iou_score, **kwargs
        )

    @staticmethod
    def _get_tracks(tracks):
        
        outputs = []
        for trackid, track in tracks.items():
            if not track.lost:
                outputs.append(track.output())
        return outputs

    @staticmethod
    def preprocess_input(bboxes, class_ids, detection_scores):
        
        new_bboxes = np.array(bboxes, dtype='float')
        new_class_ids = np.array(class_ids, dtype='int')
        new_detection_scores = np.array(detection_scores)

        new_detections = list(zip(new_bboxes, new_class_ids, new_detection_scores))
        return new_detections

    def update(self, bboxes, detection_scores, class_ids):

        self.frame_count += 1

        if len(bboxes) == 0:
            lost_ids = list(self.tracks.keys())

            for track_id in lost_ids:
                self.tracks[track_id].lost += 1
                if self.tracks[track_id].lost > self.max_lost:
                    self._remove_track(track_id)

            outputs = self._get_tracks(self.tracks)
            return outputs

        detections = Tracker.preprocess_input(bboxes, class_ids, detection_scores)

        track_ids = list(self.tracks.keys())

        updated_tracks, updated_detections = [], []

        if len(track_ids):
            track_centroids = np.array([self.tracks[tid].centroid for tid in track_ids])
            detection_centroids = get_centroid(np.asarray(bboxes))

            centroid_distances = distance.cdist(track_centroids, detection_centroids)

            track_indices = np.amin(centroid_distances, axis=1).argsort()

            for idx in track_indices:
                track_id = track_ids[idx]

                remaining_detections = [
                    (i, d) for (i, d) in enumerate(centroid_distances[idx, :]) if i not in updated_detections]

                if len(remaining_detections):
                    detection_idx, detection_distance = min(remaining_detections, key=lambda x: x[1])
                    bbox, class_id, confidence = detections[detection_idx]
                    self._update_track(track_id, self.frame_count, bbox, confidence, class_id=class_id)
                    updated_detections.append(detection_idx)
                    updated_tracks.append(track_id)

                if len(updated_tracks) == 0 or track_id is not updated_tracks[-1]:
                    self.tracks[track_id].lost += 1
                    if self.tracks[track_id].lost > self.max_lost:
                        self._remove_track(track_id)

        for i, (bbox, class_id, confidence) in enumerate(detections):
            if i not in updated_detections:
                self._add_track(self.frame_count, bbox, confidence, class_id=class_id)

        outputs = self._get_tracks(self.tracks)
        return outputs
