import os, shutil

path = 'C:/Users/admin/Desktop/megaage_asian/train/'
os.chdir('C:/Users/admin/Desktop/final/age_gender_estimation-master (2)')
#print(os.getcwd)

list = []
a = ['abc.jpg', 'efg.jpg']
#print(a[0][:-4])
path = 'C:/Users/admin/Desktop/megaage_asian/train/'

imgs = os.listdir(path)
#print(imgs[-1][-5])

#for i in range(len(imgs)):
# for i in imgs:
#     print(i)
#     i = i.split('.')[0].replace()
#     list.append(i)
#     print(len(list))
    #print(list[0][:-4])
    # print(list)
    # list.sort()
    # print(list)


path = 'C:/Users/admin/Desktop/megaage_asian/train/'

imgs = os.listdir(path)
numbers = []
for i in imgs:
    numbers.appens(int(i.split('.')[0]))
    numbers.sort()
for i in numbers:
    print(i)



