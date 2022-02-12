import os

path = 'C:/Users/admin/Desktop/last_images/megaage_asian/train/'

numbers = []
for img in os.listdir(path):
    numbers.append(int(img.split('.')[0]))
numbers.sort()
# print(numbers)

# 나이경로
with open('C:/Users/admin/Desktop/last_images/megaage_asian/list/train_age.txt', 'r') as f:
    idx_age_list = []
    i = 0
    for idx, age in enumerate(f):
        age = age.replace('/n', '')
        idx_age_list.append((idx,age))
        # print(idx)
    for pair in zip(numbers,idx_age_list):
        pair[1][1].replace()
        os.rename(path+str(pair[0])+'.jpg', path+str(pair[1][1].replace)+'_'+str(i)+'.jpg')
        # i =+ 1

    