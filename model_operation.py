import numpy as np
import cv2
import os
from tools import Bboxes2JSON,InitLabels2bgrDict,Drawing

def Training(model,train_data,validation_data=None,batch_size=1,epochs=1,step_per_epoch=1,callbacks=[]):
    def gen():yield 1
    if(type(train_data)==type(gen())):
        model.fit(train_data,
                  validation_data=validation_data,
                  epochs=epochs,
                  steps_per_epoch=step_per_epoch,
                  max_queue_size=32,
                  workers=1,
                  shuffle=False,
                  use_multiprocessing=False,
                  callbacks=callbacks)
    elif(type(train_data)==list or type(train_data)==tuple):
        model.fit(train_data,
                  validation_data=validation_data,
                  epochs=epochs,
                  callbacks=callbacks)

# def _PredictingCnfdMap(model,img):
#     orig_img_hw=np.shape(img)[:2]
#     output_hw=np.array(model.layers[0].get_input_shape_at(0)[1:3])
#     _img=cv2.resize(img,(output_hw[1],output_hw[0]))
#     _img=_img/255
#     _img=np.array([_img])
#     pred_list=model.predict_on_batch(_img)
#     cnfd_map=[]
#     for pred_msg in pred_list:
#         pred_msg=pred_msg[0]
#         pred_cnfd=pred_msg[...,8:9]
#         pred_cls=pred_msg[...,9:]
#         output_hw=np.shape(pred_cnfd)[:2]

#         pred_cnfd=pred_cnfd*pred_cls
#         pred_cnfd=np.max(pred_cnfd,axis=-1)
        
#         # pred_cnfd=np.squeeze(pred_cnfd,axis=-1)
#         pred_cnfd=cv2.resize(pred_cnfd,(orig_img_hw[1],orig_img_hw[0]))
#         cnfd_map.append(pred_cnfd)
#     cnfd_map=np.concatenate(cnfd_map,axis=-1)
#     channel=np.shape(cnfd_map)[-1]
#     cnfd_map=np.sum(cnfd_map,axis=-1)
#     cnfd_map=cnfd_map/channel
    
#     cnfd_map=np.expand_dims(cnfd_map,axis=-1)
#     out_img=None
#     out_img=cv2.normalize(cnfd_map,out_img,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
#     out_img=cv2.applyColorMap(out_img,cv2.COLORMAP_JET)
#     return out_img

# def _PredictingCnfdMaps(model_1,model_2,labels,imgs_dir,pred_dir,printing=False,img_type="jpg"):
#     InitLabels2bgrDict(labels)
#     imgs_name=os.listdir(imgs_dir)
#     for i,img_name in enumerate(imgs_name):
#         try:
#             name,_type=img_name.split(".")
#             if(_type!=img_type):continue
#         except:continue
#         img=cv2.imread(imgs_dir+"/"+img_name)
#         out_img=PredictingCnfdMap(model_1,img)
#         pred_bboxes=Predicting(model_2,labels,img)
#         out_img=(img*0.5)+(out_img*0.5)
#         # out_img=Drawing(out_img,pred_bboxes)
#         cv2.imwrite(pred_dir+"/"+img_name,out_img)
#         if(printing==True):print(str(i)+" Predicting Done.")
#     return 

def PredictingCnfdMap(model,img):
    orig_img_hw=np.shape(img)[:2]
    output_hw=np.array(model.layers[0].get_input_shape_at(0)[1:3])
    _img=cv2.resize(img,(output_hw[1],output_hw[0]))
    _img=_img/255
    _img=np.array([_img])
    pred_list=model.predict_on_batch(_img)

    out_imgs=[]
    for pred_msg in pred_list:
        pred_msg=pred_msg[0]
        pred_cnfd=pred_msg[...,8:9]
        pred_cls=pred_msg[...,9:]
        output_hw=np.shape(pred_cnfd)[:2]

        pred_cnfd=pred_cnfd*pred_cls
        pred_cnfd=np.max(pred_cnfd,axis=-1)
        
        out_img=None
        cnfd_map=cv2.resize(pred_cnfd,(orig_img_hw[1],orig_img_hw[0]))
        out_img=cv2.normalize(cnfd_map,out_img,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
        out_img=cv2.applyColorMap(out_img,cv2.COLORMAP_JET)
        out_imgs.append(out_img)
    return out_imgs

def PredictingCnfdMaps(model_1,model_2,labels,imgs_dir,pred_dir,printing=False,img_type="jpg"):
    InitLabels2bgrDict(labels)
    imgs_name=os.listdir(imgs_dir)
    for i,img_name in enumerate(imgs_name):
        try:
            name,_type=img_name.split(".")
            if(_type!=img_type):continue
        except:continue
        img=cv2.imread(imgs_dir+"/"+img_name)
        out_imgs=PredictingCnfdMap(model_1,img)
        main_name=img_name.split(".")[0]
        for j in range(len(out_imgs)):
            out_img=(img*0.5)+(out_imgs[j]*0.5)
            cv2.imwrite(pred_dir+"/"+main_name+"_"+str(j)+".jpg",out_img)
        if(printing==True):print(str(i)+" Predicting Done.")
    return 


def Predicting(model,labels,img):
    orig_img_hw=np.shape(img)[:2]
    output_hw=np.array(model.layers[0].get_input_shape_at(0)[1:3])
    wh_ratio=np.flip(orig_img_hw/output_hw,axis=-1)
    img=cv2.resize(img,(output_hw[1],output_hw[0]))
    img=img/255
    img=np.array([img])
    pred_msg=model.predict_on_batch(img)
    if(np.shape(pred_msg)[0]==0):return np.array([])
    pred_boxes=(pred_msg[...,:4]*np.concatenate([wh_ratio,wh_ratio],axis=-1)).astype("float")
    pred_boxes=np.around(pred_boxes,decimals=1)
    pred_scores=pred_msg[...,4:5]
    pred_classes=pred_msg[...,5:]
    pred_classes=pred_classes.tolist()
    pred_classes=list(map(lambda x:[labels[int(x[0])]],pred_classes))
    pred_classes=np.array(pred_classes)
    pred_bboxes=np.concatenate([pred_boxes,pred_scores,pred_classes],axis=-1)
    return pred_bboxes

def PredictingImgs(model,labels,imgs_dir,pred_dir,drawing=False,printing=False,img_type="jpg"):
    if(drawing==True):InitLabels2bgrDict(labels)
    imgs_name=os.listdir(imgs_dir)
    for i,img_name in enumerate(imgs_name):
        try:
            name,_type=img_name.split(".")
            if(_type!=img_type):continue
        except:continue
        img=cv2.imread(imgs_dir+"/"+img_name)
        pred_bboxes=Predicting(model,labels,img)
        if(drawing==True):
            img=Drawing(img,pred_bboxes)
            cv2.imwrite(pred_dir+"/"+img_name,img)
        Bboxes2JSON(pred_bboxes,pred_dir+"/json/"+name+".json")
        if(printing==True):print(str(i)+" Predicting Done.")
    return 