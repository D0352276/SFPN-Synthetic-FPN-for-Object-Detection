import tensorflow as tf
import numpy as np
swish=tf.keras.layers.Lambda(lambda x:x*tf.math.sigmoid(x))
hard_sigmoid=tf.keras.layers.Lambda(lambda x:tf.nn.relu6(x+3.0)/6.0)
mish=tf.keras.layers.Lambda(lambda x:x*tf.math.tanh(tf.math.softplus(x)))

class ConvBN(tf.Module):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1,1),
                 padding="same",
                 bias=False,
                 use_bn=True,
                 activation=None,
                 name="convbn"):
        super(ConvBN,self).__init__()
        self._filters=filters
        self._kernel_size=kernel_size
        self._strides=strides
        self._padding=padding
        self._bias=bias
        self._use_bn=use_bn
        self._activation=activation
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._conv=tf.keras.layers.Conv2D(filters=self._filters,
                                          kernel_size=self._kernel_size,
                                          strides=self._strides,
                                          padding=self._padding,
                                          use_bias=self._bias,
                                          name=self._name+"_conv")
        if(self._use_bn==True):self._bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_bn")
        self._act=tf.keras.layers.Activation(self._activation,name=self._name+"_act")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._conv(input_ts)
        if(self._use_bn==True):x=self._bn(x)
        x=self._act(x)
        output_ts=x
        return output_ts

class DepthConvBN(tf.Module):
    def __init__(self,
                 kernel_size,
                 strides=(1,1),
                 padding="same",
                 bias=False,
                 use_bn=True,
                 activation=None,
                 name="depthconvbn"):
        super(DepthConvBN,self).__init__(name=name)
        self._kernel_size=kernel_size
        self._strides=strides
        self._padding=padding
        self._bias=bias
        self._use_bn=use_bn
        self._activation=activation
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._depthconv=tf.keras.layers.DepthwiseConv2D(self._kernel_size,
                                                        self._strides,
                                                        depth_multiplier=1,
                                                        padding=self._padding,
                                                        use_bias=self._bias,
                                                        name=self._name+"_depthconv")
        if(self._use_bn==True):self._bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_bn")
        self._act=tf.keras.layers.Activation(self._activation,name=self._name+"_act")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._depthconv(input_ts)
        if(self._use_bn==True):x=self._bn(x)
        x=self._act(x)
        output_ts=x
        return output_ts

class SeparableConvBN(tf.Module):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1,1),
                 padding="same",
                 bias=False,
                 use_bn=True,
                 activation=None,
                 name="spbconvbn"):
        super(SeparableConvBN,self).__init__(name=name)
        self._filters=filters
        self._kernel_size=kernel_size
        self._strides=strides
        self._padding=padding
        self._bias=bias
        self._use_bn=use_bn
        self._activation=activation
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._spbconv=tf.keras.layers.SeparableConv2D(self._filters,
                                                      self._kernel_size,
                                                      self._strides,
                                                      depth_multiplier=1,
                                                      padding=self._padding,
                                                      use_bias=self._bias,
                                                      name=self._name+"_spbconv")
        if(self._use_bn==True):self._bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_bn")
        self._act=tf.keras.layers.Activation(self._activation,name=self._name+"_act")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._spbconv(input_ts)
        if(self._use_bn==True):x=self._bn(x)
        x=self._act(x)
        output_ts=x
        return output_ts

class AdaptUpsample(tf.Module):
    def __init__(self,output_hw,name="adaptupsample"):
        super(AdaptUpsample,self).__init__(name=name)
        self._output_hw=output_hw
        self._name=name
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        output_ts=tf.image.resize(input_ts,self._output_hw,method=tf.image.ResizeMethod.BILINEAR)
        return output_ts

class AdaptPooling(tf.Module):
    def __init__(self,output_hw,name="adaptpooling"):
        super(AdaptPooling,self).__init__(name=name)
        self._output_hw=output_hw
        self._name=name
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        output_ts=tf.image.resize(input_ts,self._output_hw,method=tf.image.ResizeMethod.BILINEAR)
        return output_ts

class AdaptScaling(tf.Module):
    def __init__(self,output_hw,name="adaptscaling"):
        super(AdaptScaling,self).__init__(name=name)
        self._output_hw=output_hw
        self._name=name
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        output_ts=tf.image.resize(input_ts,self._output_hw,method=tf.image.ResizeMethod.BILINEAR)
        return output_ts

class InputBIFusion(tf.Module):
    def __init__(self,name="inputbufusion"):
        super(InputBIFusion,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,btm_shape,top_shape):
        btm_shape=np.array(btm_shape)
        top_shape=np.array(top_shape)
        target_shape=np.round((btm_shape+top_shape)/2)
        self._btm_down=AdaptPooling(target_shape[0:2],name=self._name+"_btm_down")
        self._top_up=AdaptUpsample(target_shape[0:2],name=self._name+"_top_up")
        self._conv=ConvBN(filters=top_shape[2],kernel_size=(3,3),activation=mish,name=self._name+"_conv")
    @tf.Module.with_name_scope
    def __call__(self,btm_ts,top_ts):
        btm_shape=btm_ts.get_shape().as_list()[1:]
        top_shape=top_ts.get_shape().as_list()[1:]
        self._Build(btm_shape,top_shape)
        btm_down=self._btm_down(btm_ts)
        top_up=self._top_up(top_ts)
        x=btm_down+top_up
        output_ts=self._conv(x)
        return output_ts

class LayerExpansion(tf.Module):
    def __init__(self,out_layers=3,name="layerexpansion"):
        super(LayerExpansion,self).__init__(name=name)
        self._out_layers=out_layers
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._bifusion_list=[]
        for i in range(self._out_layers-3):
            self._bifusion_list.append(InputBIFusion(name=self._name+"_inbifusion"+str(i)))
    @tf.Module.with_name_scope
    def __call__(self,input_ts_list):
        l1,l2,l3=input_ts_list
        if(self._out_layers==3):
            return [l1,l2,l3]
        elif(self._out_layers==5):
            l1l2=self._bifusion_list[0](l1,l2)
            l2l3=self._bifusion_list[1](l2,l3)
            return [l1,l1l2,l2,l2l3,l3]
        elif(self._out_layers==9):
            l1l2=self._bifusion_list[0](l1,l2)
            l2l3=self._bifusion_list[1](l2,l3)
            l1l1l2=self._bifusion_list[2](l1,l1l2)
            l1l2l2=self._bifusion_list[3](l1l2,l2)
            l2l2l3=self._bifusion_list[4](l2,l2l3)
            l2l3l3=self._bifusion_list[5](l2l3,l3)
            return [l1,l1l1l2,l1l2,l1l2l2,l2,l2l2l3,l2l3,l2l3l3,l3]

class BIFusion(tf.Module):
    def __init__(self,name="bufusion"):
        super(BIFusion,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,mid_shape):
        target_shape=mid_shape
        self._btm_down=AdaptPooling(target_shape[0:2],name=self._name+"_btm_down")
        self._top_up=AdaptUpsample(target_shape[0:2],name=self._name+"_top_up")
        self._conv=ConvBN(filters=target_shape[2],kernel_size=(3,3),activation=mish,name=self._name+"_conv")
    @tf.Module.with_name_scope
    def __call__(self,btm_ts,mid_ts,top_ts):
        mid_shape=mid_ts.get_shape().as_list()[1:]
        self._Build(mid_shape)
        out_ts=mid_ts
        if(btm_ts!=None):
            btm_down=self._btm_down(btm_ts)
            out_ts=out_ts+btm_down
        if(top_ts!=None):
            top_up=self._top_up(top_ts)
            out_ts=out_ts+top_up
        output_ts=self._conv(out_ts)
        return output_ts

class FusionPhase1(tf.Module):
    def __init__(self,name="fusionphase1"):
        super(FusionPhase1,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,input_ts_len):
        self._bidusion_list=[]
        for i in range(input_ts_len-1):
            self._bidusion_list.append(BIFusion(name=self._name+"_bifusion"+str(i)))
    @tf.Module.with_name_scope
    def __call__(self,input_ts_list):
        input_ts_len=len(input_ts_list)
        self._Build(input_ts_len)
        if(input_ts_len==3):
            l1,l2,l3=input_ts_list
            l2=self._bidusion_list[0](l1,l2,l3)
            return [l1,l2,l3]
        elif(input_ts_len==5):
            l1,l2,l3,l4,l5=input_ts_list
            l2=self._bidusion_list[0](l1,l2,l3)
            l4=self._bidusion_list[1](l3,l4,l5)
            return [l1,l2,l3,l4,l5]
        elif(input_ts_len==9):
            l1,l2,l3,l4,l5,l6,l7,l8,l9=input_ts_list
            l2=self._bidusion_list[0](l1,l2,l3)
            l4=self._bidusion_list[1](l3,l4,l5)
            l6=self._bidusion_list[2](l4,l6,l7)
            l8=self._bidusion_list[3](l7,l8,l9)
            return [l1,l2,l3,l4,l5,l6,l7,l8,l9]

class FusionPhase2(tf.Module):
    def __init__(self,name="fusionphase2"):
        super(FusionPhase2,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,input_ts_len):
        self._bidusion_list=[]
        for i in range(input_ts_len-1):
            self._bidusion_list.append(BIFusion(name=self._name+"_bifusion"+str(i)))
    @tf.Module.with_name_scope
    def __call__(self,input_ts_list):
        input_ts_len=len(input_ts_list)
        self._Build(input_ts_len)
        if(input_ts_len==3):
            l1,l2,l3=input_ts_list
            l1=self._bidusion_list[0](None,l1,l2)
            l3=self._bidusion_list[1](l2,l3,None)
            return [l1,l2,l3]
        elif(input_ts_len==5):
            l1,l2,l3,l4,l5=input_ts_list
            l1=self._bidusion_list[0](None,l1,l2)
            l3=self._bidusion_list[1](l2,l3,l4)
            l5=self._bidusion_list[2](l4,l5,None)
            return [l1,l2,l3,l4,l5]
        elif(input_ts_len==9):
            l1,l2,l3,l4,l5,l6,l7,l8,l9=input_ts_list
            l1=self._bidusion_list[0](None,l1,l2)
            l3=self._bidusion_list[1](l2,l3,l4)
            l5=self._bidusion_list[2](l4,l5,l6)
            l7=self._bidusion_list[3](l6,l7,l8)
            l9=self._bidusion_list[4](l8,l9,None)
            return [l1,l2,l3,l4,l5,l6,l7,l8,l9]

class CSLFPN(tf.Module):
    def __init__(self,repeat=3,name="cslfpn"):
        super(CSLFPN,self).__init__(name=name)
        self._repeat=repeat
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._fusion_phase1_list=[]
        self._fusion_phase2_list=[]
        for i in range(self._repeat):
            self._fusion_phase1_list.append(FusionPhase1(name=self._name+"_phase1_"+str(i)))
            self._fusion_phase2_list.append(FusionPhase2(name=self._name+"_phase2_"+str(i)))
    @tf.Module.with_name_scope
    def __call__(self,input_ts_list):
        out_ts_list=input_ts_list
        for i in range(self._repeat):
            last_out_ts_list=out_ts_list.copy()
            out_ts_list=self._fusion_phase1_list[i](out_ts_list)
            out_ts_list=self._fusion_phase2_list[i](out_ts_list)
            for ts_idx in range(len(out_ts_list)):
                out_ts_list[ts_idx]=out_ts_list[ts_idx]+last_out_ts_list[ts_idx]
        return out_ts_list

class VanillaFPN(tf.Module):
    def __init__(self,name="vanillafPN"):
        super(VanillaFPN,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,l1_shape,l2_shape,l3_shape,l4_shape,l5_shape):
        l1_shape=np.array(l1_shape)
        l2_shape=np.array(l2_shape)
        l3_shape=np.array(l3_shape)
        l4_shape=np.array(l4_shape)
        l5_shape=np.array(l5_shape)
        self._l2_up=AdaptUpsample(l1_shape[0:2],name=self._name+"_l2_up")
        self._l3_up=AdaptUpsample(l2_shape[0:2],name=self._name+"_l3_up")
        self._l4_up=AdaptUpsample(l3_shape[0:2],name=self._name+"_l4_up")
        self._l5_up=AdaptUpsample(l4_shape[0:2],name=self._name+"_l5_up")
        self._l1_conv=ConvBN(filters=l1_shape[2],kernel_size=(3,3),activation=mish,name=self._name+"_l1_conv")
        self._l2_conv=ConvBN(filters=l2_shape[2],kernel_size=(3,3),activation=mish,name=self._name+"_l2_conv")
        self._l3_conv=ConvBN(filters=l3_shape[2],kernel_size=(3,3),activation=mish,name=self._name+"_l3_conv")
        self._l4_conv=ConvBN(filters=l4_shape[2],kernel_size=(3,3),activation=mish,name=self._name+"_l4_conv")
        self._l5_conv=ConvBN(filters=l5_shape[2],kernel_size=(3,3),activation=mish,name=self._name+"_l5_conv")
    @tf.Module.with_name_scope
    def __call__(self,input_ts_list):
        l1,l2,l3,l4,l5=input_ts_list
        l1_shape=l1.get_shape().as_list()[1:]
        l2_shape=l2.get_shape().as_list()[1:]
        l3_shape=l3.get_shape().as_list()[1:]
        l4_shape=l4.get_shape().as_list()[1:]
        l5_shape=l5.get_shape().as_list()[1:]
        self._Build(l1_shape,l2_shape,l3_shape,l4_shape,l5_shape)

        l4=l4+self._l5_up(l5)
        l3=l3+self._l4_up(l4)
        l2=l2+self._l3_up(l3)
        l1=l1+self._l2_up(l2)

        l1=self._l1_conv(l1)
        l2=self._l2_conv(l2)
        l3=self._l3_conv(l3)
        l4=self._l4_conv(l4)
        l5=self._l5_conv(l5)
        out_ts_list=[l1,l2,l3,l4,l5]
        return out_ts_list

class InvertedResidual(tf.Module):
    def __init__(self,filters,t,kernel_size=(3,3),strides=(1,1),first_layer=False,name="invertedresidual"):
        super(InvertedResidual,self).__init__(name=name)
        self._filters=filters
        self._t=t
        self._kernel_size=kernel_size
        self._strides=strides
        self._first_layer=first_layer
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,input_channel):
        if(self._first_layer==True):input_channel=self._filters
        tchannel=int(input_channel*self._t)
        if(self._first_layer==True):
            self._convbn_1=ConvBN(tchannel,self._kernel_size,(2,2),name=self._name+"_convbn_1")
        else:
            self._convbn_1=ConvBN(tchannel,(1,1),(1,1),name=self._name+"_convbn_1")
        self._depthconv=tf.keras.layers.DepthwiseConv2D(self._kernel_size,
                                                        self._strides,
                                                        depth_multiplier=1,
                                                        padding="same",
                                                        use_bias=False,
                                                        name=self._name+"_depthconv")
        self._bn=tf.keras.layers.BatchNormalization(name=self._name+"_bn")
        self._relu6=tf.keras.layers.ReLU(max_value=6.0,name=self._name+"_relu")
        self._convbn_2=ConvBN(self._filters,(1,1),(1,1),use_relu=False,name=self._name+"_convbn_2")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        input_ch=input_ts.get_shape().as_list()[3]
        self._Build(input_ch)
        x=self._convbn_1(input_ts)
        x=self._depthconv(x)
        x=self._bn(x)
        x=self._relu6(x)
        x=self._convbn_2(x)
        if(self._strides==(1,1) and self._filters==input_ch):
            x=(input_ts+x)
        output_ts=x
        return output_ts