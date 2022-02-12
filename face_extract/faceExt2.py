from PIL import Image
import face_recognition
import os


path = 'C:/Users/admin/Desktop/imdb_crop/' 
save = 'C:/Users/admin/Desktop/imdb_young_ext_200/' 


for folder in range(1,70):
    for j in os.listdir(path+f'{folder}'+'/'):    
        # print(j)
        image = face_recognition.load_image_file(path+f'{folder}'+'/'+j)
        face_locations = face_recognition.face_locations(image)

        for face_location in face_locations:
            top, right, bottom, left = face_location

            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            pil_image = pil_image.resize([200,200])
            
            # pil_image.show()
            # print(j)
            pil_image.save(save+j)
