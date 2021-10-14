#!/usr/bin/env python3
"""
Write a class Yolo that uses the Yolo v3 algorithm to perform object detection
"""
import tensorflow.keras as K
import numpy as np


class Yolo():
    """
    Class Yolo that uses the Yolo v3 algorithm to perform object detection
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        class constructor:
            model_path is the path to where a Darknet Keras model is stored
            classes_path is the path to where the list of class names used
                for the Darknet model, listed in order of index, can be found
            class_t is a float representing the box score threshold for the
                initial filtering step
            nms_t is a float representing the IOU threshold for
                non-max suppression
            anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
                    containing all of the anchor boxes:
                outputs is the number of outputs (predictions) made
                    by the Darknet model
                anchor_boxes is the number of anchor boxes used
                    for each prediction
                2 => [anchor_box_width, anchor_box_height]
        Public instance attributes:
            model: the Darknet Keras model
            class_names: a list of the class names for the model
            class_t: the box score threshold for the initial filtering step
            nms_t: the IOU threshold for non-max suppression
            anchors: the anchor boxes
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, z):
        """
        Caculates the sigmoid function
        """
        return 1 / (1 + np.exp(-z))

    def process_outputs(self, outputs, image_size):
        """
        Process the outputs.

        Arguments:
        outputs: list of numpy.ndarrays containing the predictions
                from the Darknet model for a single image:
            Each output will have the shape (grid_height, grid_width,
                    anchor_boxes, 4 + 1 + classes)
                grid_height & grid_width => the height and width of
                    the grid used for the output
                anchor_boxes => the number of anchor boxes used
                4 => (t_x, t_y, t_w, t_h)
                1 => box_confidence
                classes => class probabilities for all classes
        image_size: numpy.ndarray containing the image’s original size
            [image_height, image_width]
        Returns:
        tuple of (boxes, box_confidences, box_class_probs):
            boxes: a list of numpy.ndarrays of shape (grid_height,
                    grid_width, anchor_boxes, 4) containing the processed
                        boundary boxes for each output, respectively:
                4 => (x1, y1, x2, y2)
                (x1, y1, x2, y2) should represent the boundary
                    box relative to original image
            box_confidences: a list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 1) containing the
                    box confidences for each output, respectively
            box_class_probs: a list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, classes)
                    containing the box’s class probabilities
                    for each output, respectively
        """
        img_height = image_size[0]
        img_width = image_size[1]
        boxes = []
        box_confidences = []
        box_class_probs = []

        for output in outputs:
            boxes.append(output[..., 0:4])
            box_confidences.append(self.sigmoid(output[..., 4, np.newaxis]))
            box_class_probs.append(self.sigmoid(output[..., 5:]))

        for i, box in enumerate(boxes):
            grid_height, grid_width, anchor_boxes, _ = box.shape

            c = np.zeros((grid_height, grid_width, anchor_boxes), dtype=int)

            idx_x = np.arange(grid_width)
            idx_x = idx_x.reshape(1, grid_width, 1)
            idx_y = np.arange(grid_height)
            idx_y = idx_y.reshape(grid_height, 1, 1)
            Cx = c + idx_x
            Cy = c + idx_y

            centx = (box[..., 0])
            centy = (box[..., 1])
            bx = (self.sigmoid(centx) + Cx) / grid_width
            by = (self.sigmoid(centy) + Cy) / grid_height

            twidth = (box[..., 2])
            theight = (box[..., 3])
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]
            bw = (pw * np.exp(twidth)) / self.model.input.shape[1].value
            bh = (ph * np.exp(theight)) / self.model.input.shape[2].value

            x1 = bx - bw / 2
            y1 = by - bh / 2
            x2 = x1 + bw
            y2 = y1 + bh

            box[..., 0] = x1 * img_width
            box[..., 1] = y1 * img_height
            box[..., 2] = x2 * img_width
            box[..., 3] = y2 * img_height

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Arguments:
        boxes: a list of numpy.ndarrays of shape (grid_height, grid_width,
            anchor_boxes, 4) containing the processed boundary boxes for
            each output, respectively
        box_confidences: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, 1) containing the processed box
            confidences for each output, respectively
        box_class_probs: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, classes) containing the processed box
            class probabilities for each output, respectively
        Return:
            Tuple of (filtered_boxes, box_classes, box_scores):
            filtered_boxes: a numpy.ndarray of shape (?, 4) containing all
                of the filtered bounding boxes:
            box_classes: a numpy.ndarray of shape (?,) containing the class
                number that each box in filtered_boxes predicts,
                respectively
            box_scores: a numpy.ndarray of shape (?) containing the box
                scores for each box in filtered_boxes, respectively
        """

        scores = []
        for box_conf, box_class_prob in zip(box_confidences, box_class_probs):
            scores.append(box_conf * box_class_prob)

        scores_list = [score.max(axis=-1) for score in scores]
        scores_list = [score.reshape(-1) for score in scores_list]
        box_scores = np.concatenate(scores_list)
        del_idx = np.where(box_scores < self.class_t)
        box_scores = np.delete(box_scores, del_idx)

        classes_list = [box.argmax(axis=3) for box in scores]
        classes_list = [box.reshape(-1) for box in classes_list]
        box_classes = np.concatenate(classes_list)
        box_classes = np.delete(box_classes, del_idx)

        box_list = [box.reshape(-1, 4) for box in boxes]
        boxes = np.concatenate(box_list, axis=0)
        filtered_boxes = np.delete(boxes, del_idx, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of
            the filtered bounding boxes:
        box_classes: a numpy.ndarray of shape (?,) containing the class
            number for the class that filtered_boxes predicts, respectively
        box_scores: a numpy.ndarray of shape (?) containing the box scores
            for each box in filtered_boxes, respectively
        Return:
            Tuple of (box_predictions, predicted_box_classes,
                predicted_box_scores):
            box_predictions: a numpy.ndarray of shape (?, 4) containing all
                of the predicted bounding boxes ordered by class and box score
            predicted_box_classes: a numpy.ndarray of shape (?,) containing
                the class number for box_predictions ordered by class and
                box score, respectively
            predicted_box_scores: a numpy.ndarray of shape (?) containing
                the box scores for box_predictions ordered by class and
                box score, respectively

        """

        idx = np.lexsort((-box_scores, box_classes))
        box_pred = np.array([filtered_boxes[i] for i in idx])
        pred_classes = np.array([box_classes[i] for i in idx])
        pred_scores = np.array([box_scores[i] for i in idx])

        _, class_counts = np.unique(pred_classes, return_counts=True)

        i = 0
        accum = 0

        for class_count in class_counts:
            while i < accum + class_count:
                j = i + 1
                while j < accum + class_count:

                    box1 = box_pred[i]
                    box2 = box_pred[j]
                    xi1 = max(box1[0], box2[0])
                    yi1 = max(box1[1], box2[1])
                    xi2 = min(box1[2], box2[2])
                    yi2 = min(box1[3], box2[3])
                    inter_area = max(yi2 - yi1, 0) * max(xi2 - xi1, 0)

                    box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
                    box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
                    union_area = box1_area + box2_area - inter_area

                    iou = inter_area / union_area

                    if iou > self.nms_t:
                        box_pred = np.delete(box_pred, j, axis=0)
                        pred_scores = np.delete(pred_scores, j, axis=0)
                        pred_classes = (np.delete(pred_classes, j, axis=0))
                        class_count -= 1
                    else:
                        j += 1
                i += 1
            accum += class_count

        return box_pred, pred_classes, pred_scores
