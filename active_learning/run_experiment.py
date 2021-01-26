from pycocotools.coco import COCO
import json
import torch
import argparse
from pathlib import Path

import numpy as np
from active_learning.uncertainty_query_methods import get_next_images_by_entropy
from main import get_args_parser,main

class COCOdataset():

    def __init__(self,dirpath,datatype):
        self.dataDir = dirpath
        self.dataType = datatype
        self.annFile = '{}annotations/{}.json'.format(self.dataDir,self.dataType)
        self.load_coco_json()

    def load_coco(self):
        self.coco = COCO(self.annFile)
        return self.coco

    def load_coco_json(self):
        with open(self.annFile) as f:
            self.coco_dataset = json.load(f)
            self.images = self.coco_dataset['images']
            self.annotations = self.coco_dataset['annotations']
            self.categories = self.coco_dataset['categories']




def get_next_samples_by_random(unlabeled_dataset, n_samples,random_state=42):
    array_to_split = np.array(unlabeled_dataset.images)
    rng = np.random.default_rng(random_state)
    rng.shuffle(array_to_split)
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


def save_iteration_coco_set_json(filename,iter_i, json_dic,load_last_train_iter=False):
    if load_last_train_iter==True:
        last_json_path = '{}{}.json'.format(filename,str(iter_i-1))
        with open(last_json_path) as f:
            last_json = json.load(f)
            json_dic=combine_with_last_iteration(json_dic,last_json)

    json_path = '{}{}.json'.format(filename,iter_i)
    with open(json_path, 'wt') as coco:
        json.dump(json_dic, coco, indent=2, sort_keys=True)


def get_last_iteration(i):
    return i


def combine_with_last_iteration(current_iter, last_iter):
    for k, v in current_iter.items():
        if k in ['images','annotations']:
            current_iter[k].extend(last_iter[k])
    return current_iter


def open_coco_dataset(coco_json_path):
    with open(coco_json_path) as f:
        coco_dataset = json.load(f)
        images = coco_dataset['images']
        annotations = coco_dataset['annotations']
        categories = coco_dataset['categories']
    return images, annotations, categories

def run_next_experiment(dirpath,
                        original_data_source,
                        experiment_name,
                        cold_start_method_function,
                        query_method_function,
                        iter_i,
                        n_samples
                        ):

    cold_start_method = cold_start_method_function
    query_method = query_method_function

    if iter_i == 0:
        datasource = original_data_source
        unlabeled_coco_json = COCOdataset(dirpath, datasource)
        next_train_iter_images, unlabeled_pool_images = cold_start_method_function(unlabeled_dataset=unlabeled_coco_json,
                                                                          n_samples=n_samples)
        is_load_last_iter = False

    else:
        datasource = '{}_unlabeled_pool_iter_{}'.format(experiment_name, (iter_i - 1))
        unlabeled_coco_json = COCOdataset(dirpath, datasource)

        last_outputdir = '{}experiment/{}_train_iter_{}'.format(unlabeled_coco_json.dataDir, experiment_name, (iter_i-1))
        next_train_iter_images, unlabeled_pool_images = query_method_function(unlabeled_dataset=unlabeled_coco_json,n_samples=n_samples,
                                                                              last_output=last_outputdir)
        is_load_last_iter = True



    iteration_filename = '{}annotations/{}_train_iter_'.format(unlabeled_coco_json.dataDir, experiment_name)

    save_iteration_coco_set_json(iteration_filename, iter_i,
                                 json_dic={'iteration': iter_i, 'images': next_train_iter_images,
                                           'annotations': annotation_by_image_id(unlabeled_coco_json.annotations,
                                                                                 next_train_iter_images),
                                           'categories': unlabeled_coco_json.categories},
                                 load_last_train_iter=is_load_last_iter)

    unlabeled_filename = '{}annotations/{}_unlabeled_pool_iter_'.format(unlabeled_coco_json.dataDir,
                                                                        experiment_name)

    save_iteration_coco_set_json(unlabeled_filename, iter_i,
                                 json_dic={'iteration': iter_i, 'images': unlabeled_pool_images,
                                           'annotations': annotation_by_image_id(unlabeled_coco_json.annotations,
                                                                                 unlabeled_pool_images),
                                           'categories': unlabeled_coco_json.categories},
                                 load_last_train_iter=False)

    dataset_json_train = '{}{}.json'.format(iteration_filename,iter_i)
    output_model_dir =  '{}experiment/{}_train_iter_{}'.format(unlabeled_coco_json.dataDir, experiment_name,iter_i)
    return dataset_json_train,output_model_dir



if __name__ == '__main__':
    torch.set_num_threads(1)
    dirpath = '/Users/uri/Documents/Uri/school/Thesis/Object_Detection_AL/scripts/balloon/'
    experiment_name = 'entropy_query_baloon_v0'
    n_samples = 10
    original_data_source = 'custom_train_full'
    cold_start_method_function = get_next_samples_by_random
    query_method_function = get_next_images_by_entropy #get_next_samples_by_random  #get_next_images_by_entropy
    for i in range(3,6):
        dataset_json_train, output_model_dir = run_next_experiment(dirpath,
                                                                   original_data_source,
                                                                   experiment_name,
                                                                   cold_start_method_function,
                                                                   query_method_function,
                                                                   iter_i=i,
                                                                   n_samples=10)


        args_to_argdict = {'dataset_file':'custom',
                        'coco_path':dirpath,
                        'output_dir':output_model_dir,
                        'resume':'/Users/uri/Documents/Uri/school/Thesis/Object_Detection_AL/src/detr-r50_no-class-head.pth', #checkpoint model
                        'dataset_json_train':dataset_json_train,
                        'num_classes':1,
                        'epochs':5,
                        'device':'cpu'}



        parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
        args = parser.parse_args()
        arg_dict = vars(args)

        for key,value in args_to_argdict.items():
            arg_dict[key]= value
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        torch.set_grad_enabled(True)
        main(args)





