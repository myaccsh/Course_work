import keras
import os
import random
import itertools
import numpy as np
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from keras.utils import to_categorical
from keras.utils.data_utils import OrderedEnqueuer

class DataGenerator(keras.utils.Sequence):
    def __init__(self, src_dir, img_shape=(416,416), batch_size=4, shuffle=True):
        self.images = []
        self.labels = []
        self.src_dir = src_dir
        self.src_dir = os.path.join(self.src_dir, '')
        self.amount_of_slashes = self.src_dir.count('/')
        self.img_shape = img_shape
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()

        #/batch1/
        #       /images/1.jpg
        #       /labels/1.xml

        # /batch1/
        #       /folder/1.jpg
        #       /folder/1.xml

    def __len__(self):
        return int(len(self.images) / float(self.batch_size))

    def on_epoch_end(self):
        self.images = []
        self.labels = []

        for folder, subs, files in os.walk(self.src_dir):
            checkfiles = [f for f in os.listdir(folder) if f.endswith(".jpg")]
            if len(checkfiles) > 0:
                if self.shuffle:
                    rnd = random.random() * 10000
                    random.Random(rnd).shuffle(checkfiles)
                for f in checkfiles:
                    self.images.append(os.path.join(folder, f))
                    #todo - Добавить правильный путь к файлу разметки
                    self.labels.append(os.path.join(folder, f))

        if self.shuffle:
            rnd = random.random() * 10000
            random.Random(rnd).shuffle(self.images)
            random.Random(rnd).shuffle(self.labels)

        #self.images = np.array(self.images)
        #self.labels = np.array(self.labels)

#1 0.716797 0.395833 0.216406 0.147222
#0 0.687109 0.379167 0.255469 0.158333
#1 0.420312 0.395833 0.140625 0.166667



    def generate_data(self, indexs):

        images_batch = []
        labels_batch = []

        for i in indexs:
            image_name = os.path.join(self.src_dir, self.images[i])
            image = cv2.imread(image_name, cv2.IMREAD_COLOR)
            labels_strings = open(self.labels[i]).read().splitlines()
            # 1 0.716797 0.395833 0.216406 0.147222
            # cl xcenter ycenter width height
            # img.shape = H, W, C
            bboxes = []
            for line in labels_strings:
                line = line.split(" ")
                xc = line[1] * image.shape[1]
                yc = line[2] * image.shape[0]
                w = line[3] * image.shape[1]
                h = line[4] * image.shape[0]
                x1 = xc - (w / 2)
                x2 = xc + (w / 2)
                y1 = yc - (h / 2)
                y2 = yc + (h / 2)
                bboxes.append([x1, y1, x2, y2])

            ia.seed(1)

            image = ia.quokka(size=(image.shape[1], image.shape[0]))
            bbs = BoundingBoxesOnImage([
                BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]) for bbox in bboxes
            ], shape=image.shape)

            seq = iaa.Sequential([
                iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
                iaa.LinearContrast((0.75, 1.5)),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0, 0.05 * 255), per_channel=0.5),
                iaa.Fliplr(0.5),
                #todo - исследовать влияние данных аугмнентаций на ббоксы
                #iaa.Crop(percent=(0, 0.1)),
            ])

            # Augment BBs and images.
            image_aug, bboxes_aug = seq(image=image, bounding_boxes=bbs)
            labels_strings = []
            for i, _ in enumerate(bboxes_aug):
                bbox = bboxes_aug.bounding_boxes[i]
                x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
                #todo - пересчитать в относительные координаты
                #xc, yc, w, h
                #string1 = ....
                #labels_strings.append(string1)


            # seq = iaa.Sequential(
            #     [
            #         iaa.Fliplr(0.5),
            #         iaa.Crop(percent=(0, 0.1)),
            #         iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.3))),
            #         iaa.LinearContrast((0.75, 1.5)),
            #         iaa.AdditiveGaussianNoise(loc=0, scale=(0, 0.05 * 255), per_channel=0.5),
            #         iaa.Multiply((0.8, 1.2), per_channel=0.2)
            #     ], random_order=True
            # )
            #seq_det = seq.to_deterministic()
            #augmented_image = seq_det.augment_images([image])
            #augmented_image = augmented_image[0]
            # todo resize -> letterbox image
            augmented_image = cv2.resize(image_aug, self.img_shape)
            images_batch.append(augmented_image)
            labels_batch.append(to_categorical(self.labels[i], len(self.uniq_classes)))

        images_batch = np.array(images_batch)
        labels_batch = np.array(labels_batch)
        return images_batch, labels_batch

    def __getitem__(self, item):
        indexes = [i + item * self.batch_size for i in range(self.batch_size)]
        a, la = self.generate_data(indexes)
        return a, la

if __name__ == "__main__":

    src_dir = "/home/redivan/datasets/dog_breeds/images"
    train_gen = DataGenerator(src_dir, img_shape=(512,512), batch_size=16)
    enqueuer = OrderedEnqueuer(train_gen)
    enqueuer.start(workers=1, max_queue_size=4)
    output_gen = enqueuer.get()

    gen_len = len(train_gen)
    try:
        for i in range(gen_len):
            batch = next(output_gen)
            for a, la in zip(batch[0], batch[1]):
                #todo - сохранить изображение и строки с разметкой в файл
                pass
    finally:
        enqueuer.stop()