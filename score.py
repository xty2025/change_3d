#添加：童俊杰
#拟写了一个得分系统，判断预测框与实际框的平均重合度

from parameters import data_path_
import os

def predict_score():
    data_path_score=os.path.join(data_path_,'groundtruth_rect.txt')
    data_path_png=os.path.join(data_path_,'color')
    with open(data_path_score, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    index=0
    score1,score2=[],[]
    for i in os.listdir(data_path_png):
        x1_p,y1_p,x2_p,y2_p=0,0,0,0
        x1,y1,x2,y2=labels[index].split(",")
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        if x1!=0 and y1!=0 and x2!=0 and y2!=0:
            Sc_p=x2_p*y2_p
            Sc=x2*y2
            #重合框的坐标：max(x1_p,x1),max(y1_p,y1),min(x1_p+x2_p,x1+x2),min(y1_p+y2_p,y1+y2)
            if max(x1_p,x1)<x1+x2 and max(x1_p,x1)<x1_p+x2_p and max(y1_p,y1)<y1+y2 and max(y1_p,y1)<y1_p+y2_p:
                Sc_r=(min(x1_p+x2_p,x1+x2)-max(x1_p,x1))*(min(y1_p+y2_p,y1+y2)-max(y1_p,y1))
            else:
                Sc_r=0
            if Sc_p==0:
                score1.append(0)
            else:
                score1.append(Sc_r/Sc_p)
            score2.append(Sc_r/Sc)
        elif x1_p==0 and x2_p==0 and y1_p==0 and y2_p==0:
            score1.append(1)
            score2.append(1)
        else:
            score1.append(0)
            score2.append(0)
        index+=1
    ave_score1=sum(score1)/len(score1)
    ave_score2=sum(score2)/len(score2)
    final_score=(ave_score1+ave_score2)/2
    print(f"Score1={ave_score1},score2={ave_score2},final score={final_score}")

predict_score()