# notebook
## 运行流程
在mindspeed-mm中由\MindSpeed-MM\pretrain_sora.py开始，由model = SoRAModel(args.mm.model)进入SoRAModel模型，

在其中由self.predictor = PredictModel(config.predictor).get_model()确定t2v部分的PredictModel为VideoDitSparse，

接着在video_dit_sparse.py中，存在一个VideoDiTSparseBlock类，在其中进行了attention的计算，也即self.self_atten和self.cross_atten部分，二者均调用MultiHeadSparseAttentionSBH类，forward方法中先获取qkv，进而使用torch_npu.npu_fusion_attention计算注意力输出。

此处增添一个if else逻辑，如果开启了CP，则使用ringattn_context_parallel计算注意力输出，否则依然使用torch_npu.npu_fusion_attention。
## 入参部分
ringattn_context_parallel需要的入参有(q, k, v, head_num, cp_para, scale, attention_mask, self.dropout_p)

torch_npu.npu_fusion_attention原有的入参为(q, k, v, head_num=self.num_attention_heads_per_partition_per_cp, atten_mask=mask, input_layout="SBH", scale=1 / math.sqrt(self.head_dim))

对比来看，q k v head_num scale attention_mask均为已有，dropout_p在初始化的时候已经指定，只需要注意qkv的输入维度即可。cp_para需要添加，这一部分为ring attention的一些相关参数构建，是一个字典，包括['causal', 'cp_group', 'cp_size', 'rank', 'cp_global_ranks', 'cp_group_for_send_recv_overlap', 'pse', 'pse_type', 'cp_inner_ranks', 'cp_outer_ranks', 'cp_dkv_outer_ranks', 'cp_group_for_intra_window', 'cp_group_for_intra_window_send_recv_overlap', 'scheduling_info']
## 内部实现
根据mindspeed core，ringattn_context_parallel方法通过调用AttentionWithCp类实现。内含forward、backward、compute_mask方法。这些方法的实现过程中引入了RingP2P，需要添加这个类的实现。引入了causal_forward_fetch、flash_attention_with_alibi_pse等方法，需要添加这些方法的实现。这些实现绝大多数可参考mindspeed core中的ring_context_parallel.py和utils.py。
## 输出转换
验证输出维度是否一致，如不一致进行维度变换













