#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def image_normalization(
    image: np.ndarray,
    image_min: int = 0,
    image_max: int = 255,
    epsilon: float = 1e-12,
) -> np.ndarray:
    image = np.float32(image)
    image = (image - np.min(image)) * (image_max - image_min) / (
        (np.max(image) - np.min(image)) + epsilon) + image_min
    return image


def run_inference(
    onnx_session: onnxruntime.InferenceSession,
    image: np.ndarray,
) -> np.ndarray:
    # ONNX Input Size
    input_size = onnx_session.get_inputs()[0].shape
    input_width = input_size[3]
    input_height = input_size[2]

    # Pre process:Resize, BGR->RGB, Transpose, float32 cast
    input_image = cv.resize(image, dsize=(input_width, input_height))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    results = onnx_session.run(None, {input_name: input_image})

    # Post process
    image_width, image_height = image.shape[1], image.shape[0]
    for index, result in enumerate(results):
        temp = np.squeeze(result)
        temp = sigmoid(temp)
        temp = image_normalization(temp)
        temp = temp.astype(np.uint8)
        temp = cv.bitwise_not(temp)
        temp = cv.resize(temp, dsize=(image_width, image_height))

        results[index] = temp

    average_image = np.uint8(np.mean(results, axis=0))
    fuse_image = copy.deepcopy(results[index])

    return average_image, fuse_image


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--movie', type=str, default=None)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument(
        '--model',
        type=str,
        default='model/LDC_640x360.onnx',
    )

    args = parser.parse_args()
    model_path = args.model
    image_path = args.image

    # Initialize video capture
    cap = None
    if image_path is None:
        cap_device = args.device
        if args.movie is not None:
            cap_device = args.movie
        cap = cv.VideoCapture(cap_device)

    # Load model
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=[
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )

    while True and image_path is None:
        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # Inference execution
        average_image, fuse_image = run_inference(
            onnx_session,
            frame,
        )

        elapsed_time = time.time() - start_time

        # Draw Inference elapsed time
        cv.putText(
            debug_image,
            "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('LDC Input', debug_image)
        cv.imshow('LDC Output(Average)', average_image)
        cv.imshow('LDC Output(Fuse)', fuse_image)

    if image_path is not None:
        start_time = time.time()

        # Read image
        image = cv.imread(image_path)
        # Inference execution
        average_image, fuse_image = run_inference(
            onnx_session,
            image,
        )

        elapsed_time = time.time() - start_time

        print("Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms")

        cv.imwrite('average_image.png', average_image)
        cv.imwrite('fuse_image.png', fuse_image)

    if cap is not None:
        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
