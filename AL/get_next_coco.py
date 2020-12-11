import pandas as pd
from sklearn.model_selection import train_test_split
import json
import numpy as np


def split_set(images, size):
    # image_train_set, image_test_set = train_test_split(images, train_size=size)
    return image_train_set, image_test_set


def get_next_samples_by_random(images, n_samples):
    array_to_split = np.array(images)
    np.random.shuffle(array_to_split)
    train, test = array_to_split[:n_samples], array_to_split[n_samples:]
    return train.tolist(), test.tolist()


def annotation_by_image_id(annotations, images):
    image_ids = [img['id'] for img in images]
    filter_annotations = list(filter(lambda d: d['image_id'] in image_ids, annotations))
    return filter_annotations


def save_coco_set_json(filename, info, licenses, images, annotations, categories):
    with open(filename, 'wt') as coco:
        json.dump({'info': info, 'licenses': licenses, 'images': images,
                   'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)


def save_iteration_coco_set_json(filename, **kwargs):
    with open(filename, 'wt') as coco:
        json.dump(kwargs, coco, indent=2, sort_keys=True)


def get_last_iteration(i):
    return i


def combine_with_last_iteration(current_iter, last_iter):
    current_iter.extend(last_iter)
    return current_iter


def open_coco_dataset(coco_json_path):
    with open(coco_json_path) as f:
        coco_dataset = json.load(f)
        images = coco_dataset['images']
        annotations = coco_dataset['annotations']
        categories = coco_dataset['categories']
    return images, annotations, categories


baloon_dataset = '/Users/uri/Documents/Uri/school/Thesis/Object_Detection_AL/scripts/balloon/annotations/custom_train_full.json'

samples_size = 10

for i in range(0, 6):
    unlabeled_set_file = './split_test_folder/ballon_unlabeled_set.json'
    if i == 0:
        images, annotations, categories = open_coco_dataset(baloon_dataset)
        last_image_train_set = []

    else:
        images, _, categories = open_coco_dataset(unlabeled_set_file)
        ## tmp
        last_image_train_set = get_last_iteration(image_train_set)
        ##
    iteration_file = './split_test_folder/ballon_train_iteration_{}_set.json'.format(i)

    image_train_set, image_validation_set = get_next_samples_by_random(images, n_samples=samples_size)

    image_train_set = combine_with_last_iteration(image_train_set, last_image_train_set)

    print('iter:{} , size:{}'.format(i, len(image_train_set)))
    # save iteration
    save_iteration_coco_set_json(iteration_file, iteration=i, images=image_train_set,
                                 annotations=annotation_by_image_id(annotations, image_train_set),
                                 categories=categories)

    # save new_unlabeled set
    save_iteration_coco_set_json(unlabeled_set_file, iteration=i, images=image_validation_set,
                                 annotations=annotation_by_image_id(annotations, image_validation_set),
                                 categories=categories)
