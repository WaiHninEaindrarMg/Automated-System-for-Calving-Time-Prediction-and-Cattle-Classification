import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def get_centroid(bboxes):
    
    one_bbox = False
    if len(bboxes.shape) == 1:
        one_bbox = True
        bboxes = bboxes[None, :]

    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    w, h = bboxes[:, 2], bboxes[:, 3]

    xc = xmin + 0.5*w
    yc = ymin + 0.5*h

    x = np.hstack([xc[:, None], yc[:, None]])

    if one_bbox:
        x = x.flatten()
    return x


def iou(bbox1, bbox2):
    
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1), (x0_2, y0_2, x1_2, y1_2) = bbox1, bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0.0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    iou_ = size_intersection / size_union

    return iou_


def iou_xywh(bbox1, bbox2):
    
    bbox1 = bbox1[0], bbox1[1], bbox1[0]+bbox1[2], bbox1[1]+bbox1[3]
    bbox2 = bbox2[0], bbox2[1], bbox2[0]+bbox2[2], bbox2[1]+bbox2[3]

    iou_ = iou(bbox1, bbox2)

    return iou_


def xyxy2xywh(xyxy):
    

    if len(xyxy.shape) == 2:
        w, h = xyxy[:, 2] - xyxy[:, 0] + 1, xyxy[:, 3] - xyxy[:, 1] + 1
        xywh = np.concatenate((xyxy[:, 0:2], w[:, None], h[:, None]), axis=1)
        return xywh.astype("int")
    elif len(xyxy.shape) == 1:
        (left, top, right, bottom) = xyxy
        width = right - left + 1
        height = bottom - top + 1
        return np.array([left, top, width, height]).astype('int')
    else:
        raise ValueError("Input shape not compatible.")


def xywh2xyxy(xywh):

    if len(xywh.shape) == 2:
        x = xywh[:, 0] + xywh[:, 2]
        y = xywh[:, 1] + xywh[:, 3]
        xyxy = np.concatenate((xywh[:, 0:2], x[:, None], y[:, None]), axis=1).astype('int')
        return xyxy
    if len(xywh.shape) == 1:
        x, y, w, h = xywh
        xr = x + w
        yb = y + h
        return np.array([x, y, xr, yb]).astype('int')


def midwh2xywh(midwh):
    
    if len(midwh.shape) == 2:
        xymin = midwh[:, 0:2] - midwh[:, 2:] * 0.5
        wh = midwh[:, 2:]
        xywh = np.concatenate([xymin, wh], axis=1).astype('int')
        return xywh
    if len(midwh.shape) == 1:
        xmid, ymid, w, h = midwh
        xywh = np.array([xmid-w*0.5, ymid-h*0.5, w, h]).astype('int')
        return xywh


def intersection_complement_indices(big_set_indices, small_set_indices):
   
    assert big_set_indices.shape[0] >= small_set_indices.shape[1]
    n = len(big_set_indices)
    mask = np.ones((n,), dtype=bool)
    mask[small_set_indices] = False
    intersection_complement = big_set_indices[mask]
    return intersection_complement


def nms(boxes, scores, overlapThresh, classes=None):
    
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    if scores.dtype.kind == "i":
        scores = scores.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    if classes is not None:
        return boxes[pick], scores[pick], classes[pick]
    else:
        return boxes[pick], scores[pick]

def draw_tracks_identify(image, pil_image, tracks, label):

    print(f"Label : {label}")

    count = 0
    for trk in tracks:
        trk_id = trk[1]
        xmin = trk[2]
        ymin = trk[3]
        width = trk[4]
        height = trk[5]

        # Calculate the center coordinates of the bounding box
        cx = int((xmin + width) / 2)
        cy = int((ymin + height) / 2)

        if count < len(label):
            track_id_text = "{}".format(trk_id + 1)
            identification_text = label[count]
            print("Track ID:", track_id_text)
            print("Identification:", identification_text)


        else:
            print("Count is out of range for the label string.")

        count += 1

        t_size_track_id = cv.getTextSize(str(track_id_text), cv.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        t_size_identification = cv.getTextSize(str(identification_text), cv.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]


        # Move the track ID text and identification text to the center of the bounding box
        tx_track_id = cx - t_size_track_id[0] // 2
        ty_track_id = cy + t_size_track_id[1] // 2

        tx_identification = cx - t_size_identification[0] 
        ty_identification = cy + t_size_identification[1] 

        # Draw the bounding box
        cv.rectangle(image, (int(xmin), int(ymin)), (int(width), int(height)), (0, 0, 255), 2)


        # Draw the track ID text in white
        cv.putText(image, str(track_id_text), (tx_track_id, ty_track_id), cv.FONT_HERSHEY_SIMPLEX, 1.0, [255, 255, 255], 2)

        tx_identification = int(xmin)  # Adjust this position as needed
        ty_identification = int(ymin) # Adjust this position as needed

        # Draw the identification text in red
        cv.putText(image, str(identification_text), (tx_identification, ty_identification), cv.FONT_HERSHEY_SIMPLEX, 1.0, [0, 255, 0], 2)

    # # Overlay red color mask onto the image
    alpha = 0.3  # Set transparency level

    # Convert the PIL image to a NumPy array
    numpy_image = np.array(pil_image)

    # Apply addWeighted function
    blended_image = cv.addWeighted(image, 1, numpy_image, alpha, 0)

    return blended_image

if __name__ == '__main__':
    bb = np.random.random_integers(0, 100, size=(20,)).reshape((5, 4))
    c = get_centroid(bb)
    print(bb, c)
    
    bb2 = np.array([1, 2, 3, 4])
    c2 = get_centroid(bb2)
    print(bb2, c2)

    data = {
            0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus',
            7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike',
            15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'
        }
    dict2jsonfile(data, '../../examples/pretrained_models/caffemodel_weights/ssd_mobilenet_caffe_names.json')

