import numpy as np
from pycocotools.coco import COCO
import torchvision.transforms as T
from active_learning.load_checkpoint import  load_model , get_transform
from scipy.stats import entropy
from PIL import Image
from pathlib import Path
# root = Path(dirpath)
# print((root/'train2017'))

# from torchvision.models._utils import IntermediateLayerGetter
# backbone_layer = IntermediateLayerGetter(model,return_layers  = {'backbone': "0"})



def get_next_images_by_entropy(unlabeled_dataset,n_samples,last_output):
    coco = unlabeled_dataset.load_coco()
    ids = np.array(sorted(coco.imgs.keys()))  # ids  [0]
    query_measurement = np.empty(len(ids))

    model_name =(Path(last_output)/'checkpoint.pth')
    model = load_model(model_name)
    transform = get_transform()
    print('Entropy query start')
    for i ,img_id in enumerate(ids):
        path = '%s%s/%s' %(unlabeled_dataset.dataDir, 'train2017',coco.loadImgs(int(img_id))[0]['file_name'])
        img = transform(Image.open(path)).unsqueeze(0)
        outputs = model(img)
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values
        query_measurement[i] =entropy(keep)
    index_to_query = np.argpartition(query_measurement, query_measurement.size - n_samples, axis=None)[-n_samples:]
    to_train = [img for key,img in coco.imgs.items() if img['id'] in ids[index_to_query] ]
    to_unlabeled = [img for key,img in coco.imgs.items() if img['id'] not in ids[index_to_query] ]
    return to_train,to_unlabeled