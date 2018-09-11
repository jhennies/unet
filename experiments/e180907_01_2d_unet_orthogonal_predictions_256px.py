
from data import trainGenerator, testGenerator, saveResult
from model import unet, ModelCheckpoint
import os

experiment_name = 'e180907_01_2d_unet_orthogonal_predictions_256px'
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


# # XY
# myGene = trainGenerator(
#     2, os.path.join(dataset_folder, 'em_gt', 'crop_256'), 'raw_xy_01_10', 'mem_gt_xy_01_10',
#     data_gen_args, save_to_dir=None, target_size=(256, 256)
# )
#
# model = unet(input_size=(256, 256, 1),
#              pretrained_weights=None)
# model_checkpoint = ModelCheckpoint(
#     os.path.join(experiment_folder, 'unet_membrane_xy.h5'), monitor='loss', verbose=1, save_best_only=True
# )
# model.fit_generator(myGene, steps_per_epoch=2000, epochs=5, callbacks=[model_checkpoint])
#
# # XZ
# myGene = trainGenerator(
#     2, os.path.join(dataset_folder, 'em_gt', 'crop_256', 'xz'), 'raw_xz_01_10', 'mem_gt_xz_01_10',
#     data_gen_args, save_to_dir=None, target_size=(64, 256)
# )
#
# model = unet(input_size=(64, 256, 1),
#              pretrained_weights=None)
# model_checkpoint = ModelCheckpoint(
#     os.path.join(experiment_folder, 'unet_membrane_xz.h5'), monitor='loss', verbose=1, save_best_only=True
# )
# model.fit_generator(myGene, steps_per_epoch=2000, epochs=5, callbacks=[model_checkpoint])


# # ZY
# myGene = trainGenerator(
#     2, os.path.join(dataset_folder, 'em_gt', 'crop_256', 'zy'), 'raw_zy_01_10', 'mem_gt_zy_01_10',
#     data_gen_args, save_to_dir=None, target_size=(256, 64)
# )
#
# model = unet(input_size=(256, 64, 1),
#              pretrained_weights=None)
# model_checkpoint = ModelCheckpoint(
#     os.path.join(experiment_folder, 'unet_membrane_zy.h5'), monitor='loss', verbose=1, save_best_only=True
# )
# model.fit_generator(myGene, steps_per_epoch=2000, epochs=5, callbacks=[model_checkpoint])


# # Predict XY
# testGene = testGenerator(os.path.join(dataset_folder, 'em_test/crop_256/raw_xy_10/'), num_image=64, target_size=(256, 256))
# model = unet(input_size=(256, 256, 1))
# model.load_weights(os.path.join(experiment_folder, 'unet_membrane_xy_10.h5'))
# results = model.predict_generator(testGene, 64, verbose=1)
# if not os.path.exists(os.path.join(experiment_folder, 'result_unet_xy_10')):
#     os.mkdir(os.path.join(experiment_folder, 'result_unet_xy_10'))
# saveResult(os.path.join(experiment_folder, 'result_unet_xy_10'), results)

# # Predict XZ
# testGene = testGenerator(os.path.join(dataset_folder, 'em_test/crop_256/raw_xz_10/'),
#                          num_image=256, target_size=(64, 256), filename='raw_10_{:04d}.tif')
# model = unet(input_size=(64, 256, 1))
# model.load_weights(os.path.join(experiment_folder, 'unet_membrane_xz.h5'))
# results = model.predict_generator(testGene, 256, verbose=1)
# if not os.path.exists(os.path.join(experiment_folder, 'result_unet_xz_10')):
#     os.mkdir(os.path.join(experiment_folder, 'result_unet_xz_10'))
# saveResult(os.path.join(experiment_folder, 'result_unet_xz_10'), results)

# Predict ZY
testGene = testGenerator(os.path.join(dataset_folder, 'em_test/crop_256/raw_zy_10/'),
                         num_image=256, target_size=(256, 64), filename='raw_10_{:04d}.tif')
model = unet(input_size=(256, 64, 1))
model.load_weights(os.path.join(experiment_folder, 'unet_membrane_zy.h5'))
results = model.predict_generator(testGene, 256, verbose=1)
if not os.path.exists(os.path.join(experiment_folder, 'result_unet_zy_10')):
    os.mkdir(os.path.join(experiment_folder, 'result_unet_zy_10'))
saveResult(os.path.join(experiment_folder, 'result_unet_zy_10'), results)
