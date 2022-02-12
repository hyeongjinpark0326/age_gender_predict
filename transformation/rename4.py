# 텍스트에 저장된 정보와 파일 이름과 비교하여 일치하면 이름을 바꾸는 함수
import shutil, os

path = 'C:/Users/admin/Desktop/final/faces/megaage_asian/train/'
list = []

with open('C:/Users/admin/Desktop/final/faces/megaage_asian/list/train.txt', 'r') as f:
    # print(f.readline()) # 0,18
    
    # 텍스트 파일 읽기
    for line in f:
        # print(line.rstrip('\n')) # 000, 000
        print(line.rstrip('\n').split(',')[0]) # 인덱스 0~ 출발
        pass
    # 이미지 파일 읽기
    # for file_name in os.listdir(path):
    #     # print(file_name)
    #     list.append(int(file_name.split('.')[0]))
    #     list.sort()
    # print(list[0])
    #print(line)

    # if list[idx] == idx :
    #         print(list[idx])
    # print(list)
    # print(path+file_name)
    
    # os.rename(path+file_name, path+age+'_'+str(idx)+'.jpg')   




    # for idx, age in enumerate(f):
    #     pass
    #     # age = age.rstrip('\n')
    #     # age = (age)
    #     # print(idx)
    #     # print(age) # age는 rstrip 해야함
    # for file_name in os.listdir(path):
    #     # print(file_name)
    #     list.append(int(file_name.split('.')[0]))
    #     list.sort()
    # print(idx[1])
    
        # if list[idx] == idx :
        #     print(list[idx])
    # print(list)
    # print(path+file_name)
    
        # os.rename(path+file_name, path+age+'_'+str(idx)+'.jpg')
    
    # if list[idx] == idx :
    #     print(idx)
        #     os.rename(path+file_name, path+age+'_'+str(idx)+'.jpg')
        # else:
        #     passs
   
    


        # for i in os.listdir(path):
        #     #print(idx)
        #     list.append(int(i.split('.')[0]))
        #     list.sort()
        #     if idx == i.split('.')[0]:
        #         os.rename(path+i, path+f'{age}_{i[-4:]}.jpg')
        #     elif idx != i.split('.')[0]:
        #         pass
        #     else:
        #         pass



        # for l in list:
        #     # print(l)
        

            # for j in range(len(list)):
            #     list[j] = int(list[j])
            # list.sort()
            #print(list)

            #print(i.split('.')[0])
            


