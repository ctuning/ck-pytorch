#!/usr/bin/env python3

import json
import time
import os
import shutil
import numpy as np
import torch

from imagenet_helper import (load_preprocessed_batch, image_list, class_labels,
    MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT, MODEL_DATA_LAYOUT, MODEL_COLOURS_BGR, MODEL_INPUT_DATA_TYPE, MODEL_DATA_TYPE,
    IMAGE_DIR, IMAGE_LIST_FILE, MODEL_NORMALIZE_DATA, SUBTRACT_MEAN, GIVEN_CHANNEL_MEANS, BATCH_SIZE)

TORCH_MODEL_NAME        = os.getenv('ML_TORCH_MODEL_NAME', 'resnet50')

## Writing the results out:
#
RESULTS_DIR             = os.getenv('CK_RESULTS_DIR')
FULL_REPORT             = os.getenv('CK_SILENT_MODE', '0') in ('NO', 'no', 'OFF', 'off', '0')

## Processing by batches:
#
BATCH_COUNT             = int(os.getenv('CK_BATCH_COUNT', 1))


def main():
    global BATCH_SIZE
    global BATCH_COUNT

    bg_class_offset=0

    print('Images dir: ' + IMAGE_DIR)
    print('Image list file: ' + IMAGE_LIST_FILE)
    print('Batch size: {}'.format(BATCH_SIZE))
    print('Batch count: {}'.format(BATCH_COUNT))
    print('Results dir: ' + RESULTS_DIR);
    print('Normalize: {}'.format(MODEL_NORMALIZE_DATA))
    print('Subtract mean: {}'.format(SUBTRACT_MEAN))
    print('Per-channel means to subtract: {}'.format(GIVEN_CHANNEL_MEANS))
    print("Background/unlabelled classes to skip: {}".format(bg_class_offset))

    setup_time_begin = time.time()

    # Cleanup results directory
    if os.path.isdir(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    os.mkdir(RESULTS_DIR)

    # Load the [cached] Torch model
    torchvision_version = ''    # master by default
    try:
        import torchvision
        torchvision_version = ':v' + torchvision.__version__
    except Exception:
        pass

    model = torch.hub.load('pytorch/vision' + torchvision_version, TORCH_MODEL_NAME, pretrained=True)
    model.eval()


    setup_time = time.time() - setup_time_begin

    # Run batched mode
    test_time_begin = time.time()
    image_index = 0
    total_load_time = 0
    total_classification_time = 0
    first_classification_time = 0
    images_loaded = 0

    for batch_index in range(BATCH_COUNT):
        batch_number = batch_index+1
        if FULL_REPORT or (batch_number % 10 == 0):
            print("\nBatch {} of {}".format(batch_number, BATCH_COUNT))
      
        begin_time = time.time()
        batch_data, image_index = load_preprocessed_batch(image_list, image_index)

        load_time = time.time() - begin_time
        total_load_time += load_time
        images_loaded += BATCH_SIZE
        if FULL_REPORT:
            print("Batch loaded in %fs" % (load_time))

        # Classify one batch
        begin_time = time.time()

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            batch_data = batch_data.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            batch_results = model( torch.from_numpy( batch_data ) )

        classification_time = time.time() - begin_time
        if FULL_REPORT:
            print("Batch classified in %fs" % (classification_time))

        total_classification_time += classification_time
        # Remember first batch prediction time
        if batch_index == 0:
            first_classification_time = classification_time

        # Process results
        for index_in_batch in range(BATCH_SIZE):
            softmax_vector = batch_results[index_in_batch][bg_class_offset:]    # skipping the background class on the left (if present)
            global_index = batch_index * BATCH_SIZE + index_in_batch
            res_file = os.path.join(RESULTS_DIR, image_list[global_index])
            with open(res_file + '.txt', 'w') as f:
                for prob in softmax_vector:
                    f.write('{}\n'.format(prob))
            
    test_time = time.time() - test_time_begin
 
    if BATCH_COUNT > 1:
        avg_classification_time = (total_classification_time - first_classification_time) / (images_loaded - BATCH_SIZE)
    else:
        avg_classification_time = total_classification_time / images_loaded

    avg_load_time = total_load_time / images_loaded

    # Store benchmarking results:
    output_dict = {
        'setup_time_s': setup_time,
        'test_time_s': test_time,
        'images_load_time_total_s': total_load_time,
        'images_load_time_avg_s': avg_load_time,
        'prediction_time_total_s': total_classification_time,
        'prediction_time_avg_s': avg_classification_time,

        'avg_time_ms': avg_classification_time * 1000,
        'avg_fps': 1.0 / avg_classification_time,
        'batch_time_ms': avg_classification_time * 1000 * BATCH_SIZE,
        'batch_size': BATCH_SIZE,
    }
    with open('tmp-ck-timer.json', 'w') as out_file:
        json.dump(output_dict, out_file, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
