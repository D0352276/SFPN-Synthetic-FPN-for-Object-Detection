import tensorflow as tf
from cslyolo import CSLConv,CSLHead,CSLYOLOBody,CSLLoss

def VGG16(input_ts):
    model=tf.keras.applications.VGG16(input_tensor=input_ts,include_top=False,weights="imagenet")
    l1=model.layers[-9].output
    l2=model.layers[-5].output
    l3=model.layers[-1].output
    return l1,l2,l3

def MobileNetV2(input_ts,alpha=1.0):
    model=tf.keras.applications.MobileNetV2(input_tensor=input_ts,alpha=alpha,include_top=False,weights="imagenet")
    l1=model.layers[-101].output
    l2=model.layers[-39].output
    l3=model.layers[-12].output
    return l1,l2,l3

def ResNet50(input_ts):
    model=tf.keras.applications.ResNet50(input_tensor=input_ts,include_top=True,weights="imagenet")
    l1=model.layers[-97].output
    l2=model.layers[-35].output
    l3=model.layers[-3].output
    return l1,l2,l3


def CSLYOLO(input_shape,anchors_list,labels_len,fpn_filters=64,fpn_repeat=3,out_layers=3,backbone="cslb"):
    input_ts=tf.keras.Input(shape=input_shape)
    if(backbone=="m2"):
        bacbone_outputs=MobileNetV2(input_ts)
    elif(backbone=="vgg16"):
        bacbone_outputs=VGG16(input_ts)
    elif(backbone=="res50"):
        bacbone_outputs=ResNet50(input_ts)
    body_outputs=CSLYOLOBody(fpn_filters,fpn_repeat,out_layers)(*bacbone_outputs)
    net_outputs=CSLConv(anchors_list[0:],labels_len,name="cslconv")(body_outputs[0:])

    model=tf.keras.Model(input_ts,net_outputs)
    return model

def CompileCSLYOLO(model,heads_len,whts_path=None,lr=0.0001,compile_type="train"):
    if(whts_path!=None):
        model.load_weights(whts_path)
    if(compile_type=="train"):
        losses=[CSLLoss(name="cslloss_"+str(i))() for i in range(heads_len)]
        loss_weights=[1/heads_len for i in range(heads_len)]
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                        loss=losses,
                        loss_weights=loss_weights)
    return model


def CSLYOLOHead(model,heads_len,labels_len,max_boxes_per_cls=20,score_thres=0.5,iou_thres=0.5,syn_output_layers=False,nms_type="category_nms"):
    input_ts=model.layers[0].output
    orig_hw=model.layers[0].get_input_shape_at(0)[1:3]

    heads_ts=[]
    for i in range(heads_len,0,-1):
        heads_ts.append(model.layers[-i].output)
    output_op=CSLHead(orig_hw,labels_len,max_boxes_per_cls,score_thres,iou_thres,nms_type=nms_type,syn_output_layers=syn_output_layers)(heads_ts)
    model=tf.keras.Model(input_ts,output_op)
    return model
