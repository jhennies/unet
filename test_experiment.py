
from data import trainGenerator, testGenerator, saveResult
from model import unet, ModelCheckpoint

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
    2, '/g/schwab/hennies/teaching/datasets/em_gt/', 'raw', 'mem_gt',
    data_gen_args, save_to_dir=None, target_size=(512, 512)
)

model = unet(input_size=(512, 512, 1))
model_checkpoint = ModelCheckpoint('unet_membrane.h5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(myGene, steps_per_epoch=100, epochs=3, callbacks=[model_checkpoint])

testGene = testGenerator('/g/schwab/hennies/teaching/datasets/em_test/raw/', num_image=64, target_size=(512, 512))
model = unet(input_size=(512, 512, 1))
model.load_weights('unet_membrane.h5')
results = model.predict_generator(testGene, 64, verbose=1)
saveResult('/g/schwab/hennies/teaching/datasets/em_test/result_unet/', results)
