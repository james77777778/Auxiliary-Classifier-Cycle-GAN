import random
import torch


class ImageClassPool():
    """
    This class implements an image buffer that stores previously generated
    images.

    This buffer enables us to update discriminators using a history of
    generated images rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0,
                               no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []
            self.classes = []

    def query(self, images, classes, is_train=True):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the
                   buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images, classes
        return_images = []
        return_classes = []
        for image, c in zip(images, classes):
            image = torch.unsqueeze(image.data, 0)
            c = torch.unsqueeze(c.data, 0)
            # if the buffer is not full; keep inserting current images to the
            # buffer
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                self.classes.append(c)
                return_images.append(image)
                return_classes.append(c)
            else:
                if is_train:
                    p = random.uniform(0, 1)
                else:
                    p = 0
                # by 50% chance, the buffer will return a previously stored
                # image, and insert the current image into the buffer
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp_img = self.images[random_id].clone()
                    tmp_cls = self.classes[random_id]
                    self.images[random_id] = image
                    self.classes[random_id] = c
                    return_images.append(tmp_img)
                    return_classes.append(tmp_cls)
                # by another 50% chance, the buffer will return the current
                # image
                else:
                    return_images.append(image)
                    return_classes.append(c)
        # collect all the images and return
        return_images = torch.cat(return_images, 0)
        return_classes = torch.cat(return_classes, 0)
        return return_images, return_classes
