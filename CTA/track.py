import numpy as np

class Track:

    count = 0

    metadata = dict(
        data_output_formats=['CTA_challenge', 'mot_challenge']
    )

    def __init__(
        self,
        track_id,
        frame_id,
        bbox,
        detection_confidence,
        class_id=None,
        lost=0,
        iou_score=0.,
        data_output_format='CTA_challenge',
        **kwargs
    ):
        assert data_output_format in Track.metadata['data_output_formats']
        Track.count += 1
        self.id = track_id

        self.detection_confidence_max = 0.
        self.lost = 0
        self.age = 0

        self.update(frame_id, bbox, detection_confidence, class_id=class_id, lost=lost, iou_score=iou_score, **kwargs)

        if data_output_format == 'CTA_challenge':
            self.output = self.get_CTA_challenge_format

        elif data_output_format == 'mot_challenge':
            self.output = self.get_mot_challenge_format

        else:
            raise NotImplementedError

    def update(self, frame_id, bbox, detection_confidence, class_id=None, lost=0, iou_score=0., **kwargs):
        
        self.class_id = class_id
        self.bbox = np.array(bbox)
        self.detection_confidence = detection_confidence
        self.frame_id = frame_id
        self.iou_score = iou_score

        if lost == 0:
            self.lost = 0
        else:
            self.lost += lost

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.detection_confidence_max = max(self.detection_confidence_max, detection_confidence)

        self.age += 1

    @property
    def centroid(self):
        
        return np.array((self.bbox[0]+0.5*self.bbox[2], self.bbox[1]+0.5*self.bbox[3]))

    def get_CTA_challenge_format(self):
        
        CTA_tuple = (
            self.frame_id, self.id, self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], self.detection_confidence,
            -1, -1, -1
        )
        return CTA_tuple

    def get_mot_challenge_format(self):
        
        mot_tuple = (
            self.frame_id, self.id, self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], self.detection_confidence,
            -1, -1, -1
        )
        return mot_tuple

    def predict(self):
        
        raise NotImplemented

    @staticmethod
    def print_all_track_output_formats():
        print(Track.metadata['data_output_formats'])

