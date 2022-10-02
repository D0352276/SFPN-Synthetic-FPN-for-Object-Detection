import datetime
import numpy as np
def FramePerSecond(model,input_shape,test_num=500):
    imgs=[]
    for i in range(test_num+100):
        print("Creating........."+str(i)+"th test_img")
        imgs.append(np.array([np.zeros(input_shape)]))
    tot_img=len(imgs)
    for i,img in enumerate(imgs):
        if(i==100):start=datetime.datetime.now()
        model.predict_on_batch(img)
    end=datetime.datetime.now()
    cost_seconds=(end-start).seconds
    return test_num/cost_seconds