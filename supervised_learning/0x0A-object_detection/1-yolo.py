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
