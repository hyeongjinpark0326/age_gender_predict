from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import os

from numpy.lib.shape_base import split

# categories = ["f20", "f30", "f40", "f50", "f60", "m20", "m30", "m40", "m50", "m60"]
# for c in categories:
groups_folder_path = 'C:/Users/admin/Desktop/새 폴더 (4)/50-54/'   
for file in os.listdir(groups_folder_path):

    img = load_img(groups_folder_path+file)
    data_aug_gen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=15,
                                    zoom_range=0.1,
                                   
                                    
                                    horizontal_flip=True,
                                    
                                    fill_mode='nearest'
                                    )
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
        
    i = 0

        
    for batch in data_aug_gen.flow(x, batch_size=1, save_to_dir='C:/Users/admin/Desktop/새 폴더 (4)/50/' , save_format='jpg'):
        i += 1
        if i > 1: 
            break


