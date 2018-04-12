import os
import math
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


class ImageAugmentation:
    def __init__(self):
        pass

    def _get_generator(self):
        return ImageDataGenerator(
            rotation_range=360,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=10,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True
        )

    def clean(self, folder):
        class_dirs = os.listdir(folder)
        for class_dir in sorted(class_dirs):
            file_names = os.listdir(os.path.join(folder, class_dir))
            for file_name in sorted(file_names):
                if len(file_name.split('_')) > 1:
                    file_path = os.path.join(folder, class_dir, file_name)
                    print(' Deleting {}'.format(file_path))
                    os.remove(file_path)

    def run(self, folder, class_size=2000, random_seed=42):
        gen = self._get_generator()
        class_dirs = os.listdir(folder)
        print('Augmenting images in {}'.format(folder))
        for class_dir in sorted(class_dirs):
            file_names = os.listdir(os.path.join(folder, class_dir))
            n_images = len(file_names)
            augmentation_factor = int(round(class_size / n_images))
            print(' Class {} has {} images. Augmentation factor: {}. Final size: {}'.format(class_dir, n_images, augmentation_factor, n_images * augmentation_factor))
            dest_dir = os.path.join(folder, class_dir)
            if not os.path.isdir(dest_dir):
                os.mkdir(dest_dir)
            for file_name in sorted(file_names):
                if len(file_name.split('_')) > 1:
                    print(' Skipping {}'.format(file_name))
                    continue
                print(' Generating images for {}'.format(file_name))
                img = load_img(os.path.join(folder, class_dir, file_name))
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)
                prefix = '{}_augmented'.format(os.path.splitext(file_name)[0])
                i = 0
                for batch in gen.flow(x,
                                      seed=random_seed,
                                      batch_size=1,
                                      save_to_dir=dest_dir,
                                      save_prefix=prefix,
                                      save_format='jpg'):
                    i += 1
                    if i > augmentation_factor:
                        break


augmenter = ImageAugmentation()
# augmenter.clean('Train/TrainImages')
augmenter.run('Train/TrainImages')
