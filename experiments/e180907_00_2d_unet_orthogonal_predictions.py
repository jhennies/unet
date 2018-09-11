
from data import trainGenerator, testGenerator, saveResult
from model import unet, ModelCheckpoint
import os

experiment_name = 'e180907_00_2d_unet_orthogonal_predictions'
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

#
# # XZ
# myGene = trainGenerator(
#     2, os.path.join(dataset_folder, 'em_gt'), 'raw_xz', 'mem_gt_2_xz',
#     data_gen_args, save_to_dir=None, target_size=(64, 512)
# )
#
# model = unet(input_size=(64, 512, 1),
#              pretrained_weights=None)
# model_checkpoint = ModelCheckpoint(
#     os.path.join(experiment_folder, 'unet_membrane_xz.h5'), monitor='loss', verbose=1, save_best_only=True
# )
# model.fit_generator(myGene, steps_per_epoch=2000, epochs=5, callbacks=[model_checkpoint])
#
# testGene = testGenerator(os.path.join(dataset_folder, 'em_test/raw_xz/'), num_image=512, target_size=(64, 512))
# model = unet(input_size=(64, 512, 1))
# model.load_weights(os.path.join(experiment_folder, 'unet_membrane_xz.h5'))
# results = model.predict_generator(testGene, 512, verbose=1)
# if not os.path.exists(os.path.join(experiment_folder, 'result_unet_xz')):
#     os.mkdir(os.path.join(experiment_folder, 'result_unet_xz'))
# saveResult(os.path.join(experiment_folder, 'result_unet_xz'), results)

#
# # ZY
# myGene = trainGenerator(
#     2, os.path.join(dataset_folder, 'em_gt'), 'raw_zy', 'mem_gt_2_zy',
#     data_gen_args, save_to_dir=None, target_size=(512, 64)
# )
#
# model = unet(input_size=(512, 64, 1),
#              pretrained_weights=None)
# model_checkpoint = ModelCheckpoint(
#     os.path.join(experiment_folder, 'unet_membrane_zy.h5'), monitor='loss', verbose=1, save_best_only=True
# )
# model.fit_generator(myGene, steps_per_epoch=2000, epochs=5, callbacks=[model_checkpoint])

testGene = testGenerator(os.path.join(dataset_folder, 'em_test/raw_zy/'), num_image=512, target_size=(512, 64))
model = unet(input_size=(512, 64, 1))
model.load_weights(os.path.join(experiment_folder, 'unet_membrane_zy.h5'))
results = model.predict_generator(testGene, 512, verbose=1)
if not os.path.exists(os.path.join(experiment_folder, 'result_unet_zy')):
    os.mkdir(os.path.join(experiment_folder, 'result_unet_zy'))
saveResult(os.path.join(experiment_folder, 'result_unet_zy'), results)