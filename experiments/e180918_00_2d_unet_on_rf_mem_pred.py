
from data import trainGenerator, testGenerator, saveResult
from model import unet, ModelCheckpoint
import os

experiment_name = 'e180918_00_2d_unet_on_rf_mem_pred'
experiment_folder = '/g/schwab/hennies/phd_project/image_analysis/autoseg/membrane_predictions/{}'.format(experiment_name)
if not os.path.exists(experiment_folder):
    os.mkdir(experiment_folder)
dataset_folder = '/g/schwab/hennies/phd_project/image_analysis/autoseg/membrane_predictions'

data_gen_args = dict(
    rotation_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
)

myGene = trainGenerator(
    2, os.path.join(dataset_folder, 'em_gt'), 'mem_pred', 'mem_gt_2',
    data_gen_args, save_to_dir=None, target_size=(512, 512)
)

model = unet(input_size=(512, 512, 1),
             pretrained_weights=os.path.join(experiment_folder, 'unet_membrane.h5'))
model_checkpoint = ModelCheckpoint(
    os.path.join(experiment_folder, 'unet_membrane.h5'), monitor='loss', verbose=1, save_best_only=True
)
model.fit_generator(myGene, steps_per_epoch=2000, epochs=5, callbacks=[model_checkpoint])

testGene = testGenerator(os.path.join(dataset_folder, 'em_test/mem_pred/'),
                         num_image=64, target_size=(512, 512), filename='slice_{:04d}.tif')
model = unet(input_size=(512, 512, 1))
model.load_weights(os.path.join(experiment_folder, 'unet_membrane.h5'))
results = model.predict_generator(testGene, 64, verbose=1)
if not os.path.exists(os.path.join(experiment_folder, 'result_unet')):
    os.mkdir(os.path.join(experiment_folder, 'result_unet'))
saveResult(os.path.join(experiment_folder, 'result_unet'), results)
