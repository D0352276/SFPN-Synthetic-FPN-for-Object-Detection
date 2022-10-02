import argparse
from tools import ParsingCfg
from tools import InitDataDir

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("cfg_path",help="config file path",type=str)
    parser.add_argument("-t","--train",help="training mode",action="store_true")
    parser.add_argument("-ce","--cocoevaluation",help="coco evaluation mode",action="store_true")
    parser.add_argument("-e","--evaluation",help="evaluation mode",action="store_true")
    parser.add_argument("-fe","--fpsevaluation",help="fps evaluation mode",action="store_true")
    parser.add_argument("-p","--predict",help="prediction mode",action="store_true")
    parser.add_argument("-pc","--predictcnfd",help="prediction confidence map mode",action="store_true")
    args=parser.parse_args()

    mode="train"
    cfg_path=args.cfg_path
    if(args.train==True):mode="train"
    elif(args.cocoevaluation==True):mode="cocoevaluation"
    elif(args.fpsevaluation==True):mode="fpsevaluation"
    elif(args.predict==True):mode="predict"
    elif(args.predictcnfd==True):mode="predictcnfd"

    cfg_dict=ParsingCfg(cfg_path)

    input_shape=list(map(lambda x:int(x),cfg_dict["input_shape"]))
    out_hw_list=list(map(lambda x:[int(x[0]),int(x[1])],cfg_dict["out_hw_list"]))

    heads_len=len(out_hw_list)
    backbone=cfg_dict["backbone"]
    fpn_filters=cfg_dict["fpn_filters"]
    fpn_repeat=cfg_dict["fpn_repeat"]

    anchors_list=list(map(lambda x:[[1,1],[2,2],[4,4]],out_hw_list))
    anchoors_len=len(anchors_list[0])
    labels=cfg_dict["labels"]
    labels_len=len(labels)


    if(mode=="train"):
        from create_model import CSLYOLO,CompileCSLYOLO
        from data_generator import MultiDataGenerator
        from model_operation import Training
        from evaluate import CallbackEvalFunction
        from callbacks import Stabilizer,WeightsSaver,BestWeightsSaver

        init_weight=cfg_dict.get("init_weight_path",None)
        weight_save_path=cfg_dict["weight_save_path"]
        freeze=cfg_dict.get("freeze",False)

        #Must Contains Jsons
        train_dir=cfg_dict["train_dir"]
        valid_dir=cfg_dict["valid_dir"]
        pred_dir=cfg_dict["pred_dir"]

        batch_size=int(cfg_dict["batch_size"])
        step_per_epoch=int(cfg_dict["step_per_epoch"])
        epochs_schedule=list(map(lambda x:int(x),cfg_dict["epochs_schedule"]))
        lr_schedule=cfg_dict["lr_schedule"]
        callbacks_schedule=cfg_dict["callbacks_schedule"]

        gen=MultiDataGenerator(train_dir,train_dir+"/json",input_shape[:2],out_hw_list,anchors_list,labels,batch_size=batch_size,print_bool=False)
        gen.Start()

        stabilizer=Stabilizer()
        weight_saver=WeightsSaver(weight_save_path)

        model=CSLYOLO(input_shape,anchors_list,labels_len,fpn_filters,fpn_repeat,heads_len,backbone,freeze)
        model.summary()
        model=CompileCSLYOLO(model,heads_len,whts_path=init_weight,lr=0.1)
        for i,epochs in enumerate(epochs_schedule):
            callbacks=[]
            lr=lr_schedule[i]
            for callback_name in callbacks_schedule[i]:
                if(callback_name=="stabilizer"):callbacks.append(stabilizer)
                if(callback_name=="weight_saver"):callbacks.append(weight_saver)
            model=CompileCSLYOLO(model,heads_len,whts_path=None,lr=lr,compile_type="train")
            Training(model,gen.Generator(),batch_size=batch_size,epochs=epochs,step_per_epoch=step_per_epoch,callbacks=callbacks)
        gen.Stop()
    elif(mode=="predict"):
        from create_model import CSLYOLO,CompileCSLYOLO,CSLYOLOHead
        from model_operation import PredictingImgs
        
        imgs_dir=cfg_dict["imgs_dir"]
        pred_dir=cfg_dict["pred_dir"]
        InitDataDir(pred_dir)
        weight_path=cfg_dict["weight_path"]
        max_boxes_per_cls=int(cfg_dict["max_boxes_per_cls"])
        score_thres=cfg_dict["score_thres"]
        iou_thres=cfg_dict["iou_thres"]
        nms_type=cfg_dict["nms_type"]
        drawing=cfg_dict["drawing"]
        model=CSLYOLO(input_shape,anchors_list,labels_len,fpn_filters,fpn_repeat,heads_len,backbone)
        model.summary()
        model=CompileCSLYOLO(model,heads_len,whts_path=weight_path,compile_type="predict")
        model=CSLYOLOHead(model,heads_len,labels_len,max_boxes_per_cls=max_boxes_per_cls,score_thres=score_thres,iou_thres=iou_thres,nms_type=nms_type,syn_output_layers=syn_output_layers)
        PredictingImgs(model,labels,imgs_dir,pred_dir,drawing=drawing,printing=True)
    elif(mode=="predictcnfd"):
        from create_model import CSLYOLO,CompileCSLYOLO,CSLYOLOHead
        from model_operation import PredictingCnfdMaps
        imgs_dir=cfg_dict["imgs_dir"]
        pred_dir=cfg_dict["pred_dir"]
        InitDataDir(pred_dir)
        weight_path=cfg_dict["weight_path"]
        max_boxes_per_cls=int(cfg_dict["max_boxes_per_cls"])
        score_thres=cfg_dict["score_thres"]
        iou_thres=cfg_dict["iou_thres"]
        nms_type=cfg_dict["nms_type"]
        
        model=CSLYOLO(input_shape,anchors_list,labels_len,fpn_filters,fpn_repeat,heads_len,backbone)
        model.summary()
        model_1=CompileCSLYOLO(model,heads_len,whts_path=weight_path,compile_type="predict")
        model_2=CSLYOLOHead(model_1,heads_len,labels_len,max_boxes_per_cls=max_boxes_per_cls,score_thres=score_thres,iou_thres=iou_thres,nms_type=nms_type)
        PredictingCnfdMaps(model_1,model_2,labels,imgs_dir,pred_dir,printing=True)
    elif(mode=="cocoevaluation"):
        from create_model import CSLYOLO,CompileCSLYOLO,CSLYOLOHead
        from model_operation import PredictingImgs
        from evaluate.cocoeval import COCOEval
        imgs_dir=cfg_dict["imgs_dir"]
        pred_dir=cfg_dict["pred_dir"]
        annotation_path=cfg_dict["annotation_path"]
        label2id_path=cfg_dict["label2id_path"]
        weight_path=cfg_dict["weight_path"]
        max_boxes_per_cls=int(cfg_dict["max_boxes_per_cls"])
        score_thres=cfg_dict["score_thres"]
        iou_thres=cfg_dict["iou_thres"]
        nms_type=cfg_dict["nms_type"]
        overwrite=cfg_dict["overwrite"]
        syn_output_layers=cfg_dict["syn_output_layers"]
        if(overwrite==True):
            InitDataDir(pred_dir)
            model=CSLYOLO(input_shape,anchors_list,labels_len,fpn_filters,fpn_repeat,heads_len,backbone)
            model.summary()
            model=CompileCSLYOLO(model,heads_len,whts_path=weight_path,compile_type="predict")
            model=CSLYOLOHead(model,heads_len,labels_len,max_boxes_per_cls=max_boxes_per_cls,score_thres=score_thres,iou_thres=iou_thres,nms_type=nms_type,syn_output_layers=syn_output_layers)
            PredictingImgs(model,labels,imgs_dir,pred_dir,drawing=False,printing=True)
        COCOEval(annotation_path,pred_dir+"/json",label2id_path)
    elif(mode=="fpsevaluation"):
        from create_model import CSLYOLO,CompileCSLYOLO,CSLYOLOHead
        from evaluate import FramePerSecond
        max_boxes_per_cls=int(cfg_dict["max_boxes_per_cls"])
        score_thres=cfg_dict["score_thres"]
        iou_thres=cfg_dict["iou_thres"]
        nms_type="none"
        syn_output_layers=cfg_dict["syn_output_layers"]
        model=CSLYOLO(input_shape,anchors_list,labels_len,fpn_filters,fpn_repeat,heads_len,backbone)
        model=CompileCSLYOLO(model,heads_len,compile_type="predict")
        model=CSLYOLOHead(model,heads_len,labels_len,max_boxes_per_cls=max_boxes_per_cls,score_thres=score_thres,iou_thres=iou_thres,nms_type=nms_type,syn_output_layers=syn_output_layers)
        fps=FramePerSecond(model,input_shape)
        print("FPS: "+str(fps))