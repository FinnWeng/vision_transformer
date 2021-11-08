'''
prefetchDataset shapes: {image: (1, 512, 224, 224, 3), label: (1, 512, 10)}, types: {image: tf.float32, label: tf.float32}

'''

from os import name
import ml_collections
import tensorflow as tf
import tensorflow_addons as tfa
import math

from net.vit import ViT
from dataloader import get_data_from_tfds, get_dataset_info

import training_config
import model_config




if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[1], 'GPU')
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


    # initialize dataset
    dataset = "cifar10"
    
    config = training_config.with_dataset(training_config.get_config(), dataset)
    ds_train_info = get_dataset_info(dataset, "train")
    ds_train_num_classes = ds_train_info['num_classes']
    ds_train_num_examples = ds_train_info["num_examples"]
    ds_train = get_data_from_tfds(config=config, mode='train')

    ds_val_info = get_dataset_info(dataset, "test")
    ds_val_num_classes = ds_train_info['num_classes']
    ds_val_num_examples = ds_train_info["num_examples"]
    ds_val = get_data_from_tfds(config=config, mode='test')


    one_train_data = next(ds_train.as_numpy_iterator())[0]
    print("one_train_data.shape:", one_train_data["image"].shape) # vit_model_config 
    print(one_train_data["image"].shape[1:])


    # initialize model
    vit_model_config = model_config.get_b32_config()
    print(vit_model_config )
    
    vit_model = ViT(num_classes=ds_train_num_classes, **vit_model_config)

    # this init the model and avoid manipulate weight in graph(if using resnet)
    trial_logit = vit_model(one_train_data["image"], train = True) # (512, 10) 

    # build model, expose this to show how to deal with dict as fit() input
    model_input = tf.keras.Input(shape=one_train_data["image"].shape[1:],name="image",dtype=tf.float32)

    logit = vit_model(model_input)

    prob = tf.keras.layers.Softmax(axis = -1, name = "label")(logit)

    model = tf.keras.Model(inputs = [model_input],outputs = [prob], name = "ViT_model")


    '''
    the training config is for fine tune. I use my own config instead for training purpose.
    
    '''
    # my training config:
    steps_per_epoch = ds_train_num_examples//config.batch
    validation_steps = 3
    log_dir="./tf_log/"
    total_steps = 100
    warmup_steps = 5
    base_lr = 1e-3

    # define callback 
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    save_model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./model/ViT.ckpt',
        save_weights_only= True,
        verbose=1)

    callback_list = [tensorboard_callback,save_model_callback]


    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 1e-2, decay_steps = 1000, decay_rate = 0.01, staircase=False, name=None)
    # lr_schedule = Cosine_Decay_with_Warm_up(base_lr, total_steps, warmup_steps)


    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate = base_lr), 
        loss={"label":tf.keras.losses.CategoricalCrossentropy(from_logits=False)},
        metrics={'label': 'accuracy'})

    print(model.summary())

    # import pdb
    # pdb.set_trace()

    hist = model.fit(ds_train,
                epochs=200, 
                steps_per_epoch=steps_per_epoch,
                validation_data = ds_val,
                validation_steps=3,callbacks = callback_list).history


