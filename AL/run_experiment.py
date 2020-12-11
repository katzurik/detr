from pycocotools.coco import COCO
import json
import numpy as np

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


#def get_next_iter(n_iter=0,method=None,):




##
# def split_set(images, size):
#     # image_train_set, image_test_set = train_test_split(images, train_size=size)
#     return image_train_set, image_test_set


def get_next_samples_by_random(images, n_samples,random_state=42):
    array_to_split = np.array(images)
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


def save_iteration_coco_set_json(filename,iter_i, json_dic,last_iter_json=None,load_last_train_iter=False):
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


if __name__ == '__main__':
    dirpath = '/Users/uri/Documents/Uri/school/Thesis/Object_Detection_AL/scripts/balloon/'

    #datatype = 'custom_train_full'
    #original_coco_json = COCOdataset(dirpath,datatype)
    cold_start_method = get_next_samples_by_random
    query_method = get_next_samples_by_random

    experiment_name = 'random_query_baloon_v0'
    n_iter = 10
    n_samples = 10
    iter_i=6
    if iter_i == 0:
        datasource = 'custom_train_full'
        unlabeled_coco_json = COCOdataset(dirpath, datasource)
        next_train_iter_images , unlabeled_pool_images = cold_start_method(images=unlabeled_coco_json.images,n_samples=n_samples,random_state=42)
        is_load_last_iter = False

    else :
        datasource = '{}_unlabeled_pool_iter_{}'.format(experiment_name,(iter_i-1))
        unlabeled_coco_json = COCOdataset(dirpath, datasource)
        next_train_iter_images, unlabeled_pool_images = query_method(images=unlabeled_coco_json.images,
                                                                          n_samples=n_samples, random_state=42)
        is_load_last_iter= True
        last_train_iter_filename = '{}_train_iter_{}'.format(experiment_name, (iter_i-1))
        last_json = COCOdataset(dirpath, last_train_iter_filename).coco_dataset



    iteration_filename = '{}annotations/{}_train_iter_'.format(unlabeled_coco_json.dataDir,experiment_name)

    save_iteration_coco_set_json(iteration_filename,iter_i,json_dic={'iteration':iter_i, 'images':next_train_iter_images,
                                 'annotations':annotation_by_image_id(unlabeled_coco_json.annotations, next_train_iter_images),
                                 'categories':unlabeled_coco_json.categories},
                                 load_last_train_iter=is_load_last_iter)

    unlabeled_filename = '{}annotations/{}_unlabeled_pool_iter_'.format(unlabeled_coco_json.dataDir,experiment_name)

    save_iteration_coco_set_json(unlabeled_filename,iter_i, json_dic={'iteration':iter_i, 'images':unlabeled_pool_images,
                                 'annotations':annotation_by_image_id(unlabeled_coco_json.annotations, unlabeled_pool_images),
                                 'categories':unlabeled_coco_json.categories},
                                 load_last_train_iter=False)

    print('0')




