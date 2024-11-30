# notebook
## 运行流程
在mindspeed-mm中由\MindSpeed-MM\pretrain_sora.py开始，由model = SoRAModel(args.mm.model)进入SoRAModel模型，在其中由self.predictor = PredictModel(config.predictor).get_model()确定t2v部分的PredictModel为VideoDitSparse，接着在video_dit_sparse.py中，存在一个VideoDiTSparseBlock类，在其中进行了attention的计算，也即self.self_atten和self.cross_atten部分，二者均调用MultiHeadSparseAttentionSBH类，
## 入参部分
## 内部实现
## 输出转换
