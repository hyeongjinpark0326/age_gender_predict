
import os

# 이미지 경로
path = 'C:/Users/admin/Desktop/last_images/megaage_asian/train/*.*'

# 00001.jpg
'''
    1.jpg => 18_0
    2.jp => 9_1

    labels = ..... # 정답 목록

    for idx, n in enumerate( range(1, 40000+1) ):
        path = f'C:/Users/admin/Desktop/last_images/megaage_asian/train/{n}.jpg'
        # 파일이 논재하는가?
        if  조건:
            newName = f'{labels[idx]}_{n}.jpg'
            # 이름 변경 처리
'''


    # # 나이 경로
    # idx = 0
with open('C:/Users/admin/Desktop/final/faces/megaage_asian/list/train_age.txt', 'r') as f:
    # print(f.readlines()[0]) # 18
    for idx, age in enumerate(f): # f에서 인덱스와 나이를 읽어옴
        numbers = []
        for img in os.listdir(path):
            numbers.append(int(img.split('.')[0]))
        numbers.sort()
        # print(numbers) # 1부터 시작
        # print(idx) # 0부터 시작
        idx = idx+1
        if idx == numbers: # idx가 0부터 시작해서 1을 더해줌. # 텍스트 파일의 숫자 idx와 이미지 파일의 숫자가 같으면 이름 변환
            print('일치')
                # os.rename(path+img, path+f'{age}_{idx}.png')


            #print(age.replace('/n','')[0])
        #     # print(i)
        #     print(age)
            # numbers = []
            # for i in os.listdir(path):
            #     numbers.append(int(i.split('.')[0]))
            # numbers.sort()
            # for i in numbers:
            #     # print(i) # 이 i는 이미지 이름
            #     # img = img.split('.')[0]
            #     if age == i:
            #         os.rename(path+i, path+f'{age}_{i}.png')


            # # print(path+f'{img}')
            # os.rename(path+img, path+f'{age}_{img}.png'.replace('/n',''))
                

            