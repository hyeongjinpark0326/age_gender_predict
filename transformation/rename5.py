import os, shutil


path = 'C:/Users/admin/Desktop/final/faces/megaage_asian/train/'

age = open('C:/Users/admin/Desktop/final/faces/megaage_asian/list/train_age.txt', 'r')



imgs = os.listdir(path)
numbers = []
for i in imgs:
    numbers.appens(int(i.split('.')[0]))
numbers.sort()
for i in numbers:
    print(i)