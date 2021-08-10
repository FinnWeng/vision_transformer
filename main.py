'''
prefetchDataset shapes: {image: (1, 512, 224, 224, 3), label: (1, 512, 10)}, types: {image: tf.float32, label: tf.float32}

'''

import ml_collections
import tensorflow as tf

from net.vit import ViT
from dataloader import get_data_from_tfds, get_dataset_info

import training_config
import model_config




if __name__ == "__main__":

    # initialize dataset
    dataset = "cifar10"
    batch_size = 1
    config = training_config.with_dataset(training_config.get_config(), dataset)
    num_classes = get_dataset_info(dataset, "train")['num_classes']
    ds_train = get_data_from_tfds(config=config, mode='train')

    one_train_data = next(ds_train.as_numpy_iterator())
    print("one_train_data.shape:", one_train_data["image"].shape) # vit_model_config 


    # initialize model
    model_name = 'ViT-B_32'
    vit_model_config = model_config.get_b32_config()
    print(vit_model_config )
    
    model = ViT(num_classes=num_classes, **vit_model_config)

    # this init the model and avoid manipulate weight in graph

    model(one_train_data["image"], train = True)


