import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_transformer import MultiHeadAttention, PositionwiseFeedForward
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.utils.comm import all_gather_with_grad, concat_all_gather, get_rank
from .utils_motifs import rel_vectors, obj_edge_vectors, to_onehot, nms_overlaps, encode_box_info 
from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.modeling.make_layers import make_fc
import logging

class sec_branch(nn.Module):
    def __init__(self, config, in_channels):
        super(sec_branch, self).__init__()

        self.logger = logging.getLogger(__name__)
        embed_dim = config.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        roi_dim = config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        inner_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM

        num_head = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        dropout_rate = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        rel_layer = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER
        
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'
        
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES
        
        # *********************************** init bert model ***********************************
        from transformers import BertTokenizer,BertModel
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.bert_encode.pooler=None
        bert_cfg=self.bert_encoder.config
        
        self.img_cls=nn.Parameter(torch.randn(roi_dim))
        self.img_proj = nn.Sequential(
            nn.Linear(roi_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, bert_cfg.hidden_size)
        )

        statistics = get_dataset_statistics(config)

        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        
        self.cross_attention = nn.ModuleList([
            nn.ModuleList([
                # image-text cross transformer
                nn.LayerNorm(bert_cfg.hidden_size),
                nn.MultiheadAttention(bert_cfg.hidden_size, num_head,
                                      dropout_rate, batch_first=True),
                nn.LayerNorm(bert_cfg.hidden_size),
                MLP(bert_cfg.hidden_size,hidden_dim,bert_cfg.hidden_size,2),
                # text-image cross attention 
                nn.LayerNorm(bert_cfg.hidden_size),
                nn.MultiheadAttention(bert_cfg.hidden_size, num_head,
                                      dropout_rate, batch_first=True),
                nn.LayerNorm(bert_cfg.hidden_size),
                MLP(bert_cfg.hidden_size,hidden_dim,bert_cfg.hidden_size,2),
                
            ]) for _ in range(rel_layer)
        ])
        # image concate text 
        self.img_text_proj=MLP(bert_cfg.hidden_size,hidden_dim,bert_cfg.hidden_size,2)
        
        self.mask_to_rel=MLP(bert_cfg.hidden_size,hidden_dim,self.num_rel_cls,1)


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        current_device,add_losses=torch.device(f'cuda:{torch.cuda.current_device()}'),dict()
        
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)
        
        # refine object labels
        entity_dists, entity_preds = self.refine_obj_labels(roi_features, proposals)
        ##### 

        entity_dists = entity_dists.split(num_objs, dim=0)
        splited_obj_ori_preds = entity_preds.split(num_objs, dim=0)
        splited_roi_features = roi_features.split(num_objs, dim=0)
        split_union_features = union_features.split(num_rels, dim=0)
        
        # ************************************************ bert encode relationship **********************************************************************
        rel_tokenizer=self.tokenizer(self.rel_classes, add_special_tokens=True, padding=True, return_tensors='pt').to(current_device)
        rel_encode_states=self.bert_encoder(**rel_tokenizer).last_hidden_state
        encode_rel_cls=rel_encode_states[:,0]
        # ************************************************************************************************************************************************************
        rel_dists = []
        for batch_idx,proposal in enumerate(proposals):
            batch_obj_preds = splited_obj_ori_preds[batch_idx]  # (num_objs)
            batch_roi_feature = splited_roi_features[batch_idx] # (num_objs,roi_dim)
            batch_rel_pair_idx = rel_pair_idxs[batch_idx]
            batch_union_feature = split_union_features[batch_idx]

            if batch_rel_pair_idx.shape[0] == 0:
                if self.logger is not None:
                    self.logger.warning('No Graph Detected ....')
                else:
                    print(
                        f'{time.strftime("%Y-%m-%d %H:%M:%S")} maskrcnn_benchmark Warning: No Graph Detected ....')
                continue

            head_idx, tail_idx = batch_rel_pair_idx[:,
                                                    0], batch_rel_pair_idx[:, 1]
            head_obj_pre, tail_obj_pre = batch_obj_preds[head_idx], batch_obj_preds[tail_idx]
            head_obj_feature, tail_obj_feature = batch_roi_feature[head_idx], batch_roi_feature[tail_idx]

            rel_prompts = []
            for idx, (head_obj, tail_obj) in enumerate(zip(head_obj_pre, tail_obj_pre)):
                rel_prompt = f"Based on the above visual areas, the relationship between {self.obj_classes[head_obj]} and {self.obj_classes[tail_obj]} is [MASK]"
                rel_prompts.append(rel_prompt)
            
            mask_id=self.tokenizer('[MASK]',add_special_tokens=True, padding=True, return_tensors='pt').input_ids[0,1]

            # shape (num_rels,token_len,bert_dim) token[0]=[CLS]
            rel_prompt_tokenizer = self.tokenizer(
                text=rel_prompts, add_special_tokens=True, padding=True, return_tensors='pt').to(current_device)
            
            mask_row,mask_col=torch.where(rel_prompt_tokenizer.input_ids==mask_id)
            
            extended_attention_mask,head_mask,encoder_hidden_states,encoder_extended_attention_mask,past_key_values,use_cache,output_attentions,output_hidden_states,return_dict,past_key_values_length=self.prepare_bert_param(**rel_prompt_tokenizer)
            rel_prompt_embedding=self.bert_encoder.embeddings(
                        input_ids=rel_prompt_tokenizer.input_ids,
                        token_type_ids=rel_prompt_tokenizer.token_type_ids,
                        past_key_values_length=past_key_values_length)
            
            img_cls=self.img_cls.expand(batch_union_feature.shape[0],-1)
            align_vis=self.img_proj(torch.stack([img_cls,batch_union_feature,head_obj_feature,tail_obj_feature],dim=1)) # (num_rels,4,bert_dim)
            for (it_attn_ln,it_attn,it_mlp_ln,it_mlp,ti_attn_ln,ti_attn,ti_mlp_ln,ti_mlp) in self.cross_attention:
                
                it_attn_vis,_=it_attn(align_vis,rel_prompt_embedding,rel_prompt_embedding)
                ti_attn_text,_=ti_attn(rel_prompt_embedding,align_vis,align_vis)
            
                ln_it_attn_vis,ln_ti_attn_text=it_attn_ln(it_attn_vis)+align_vis,ti_attn_ln(ti_attn_text)+rel_prompt_embedding
                
                it_mlp_vis,ti_mlp_text=it_mlp(ln_it_attn_vis),ti_mlp(ln_ti_attn_text)
                
                align_vis,rel_prompt_embedding=it_mlp_ln(it_mlp_vis)+ln_it_attn_vis,ti_mlp_ln(ti_mlp_text)+ln_ti_attn_text

            bert_embedding=torch.cat([rel_prompt_embedding[:,:1,:],align_vis[:,1:,:],rel_prompt_embedding[:,1:,:]],dim=1)

            encoder_vis_text=self.bert_encoder.encoder(
                bert_embedding,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output=encoder_vis_text[0]
            
            bert_cls=sequence_output[:,0,:]
            mask_feature=sequence_output[mask_row,mask_col+align_vis.shape[1]-1,:]
            
            mask_to_rel=self.mask_to_rel(mask_feature)
            bert_cls_sim=torch.matmul(bert_cls,encode_rel_cls.permute(1,0).contiguous())
            
            if self.training:
                batch_rel_labels=rel_labels[batch_idx]
                add_losses['mask_to_rel']=add_losses.get('mask_to_rel',0.0)+F.cross_entropy(mask_to_rel,batch_rel_labels)
                add_losses['bert_cls_sim']=add_losses.get('bert_cls_sim',0.0)+F.cross_entropy(bert_cls_sim,batch_rel_labels)
            rel_dists.append(mask_to_rel+bert_cls_sim)
            
        return entity_dists, rel_dists, add_losses, dict()
      
    def prepare_bert_param(self,input_ids,inputs_embeds,past_key_values,encoder_hidden_states):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        return extended_attention_mask,head_mask,encoder_hidden_states,encoder_extended_attention_mask,past_key_values,use_cache,output_attentions,output_hidden_states,return_dict,past_key_values_length
    
        # ************************************************************************************************************************************************************


        # ************************************************************************************************************************************************************
        # ------------------------------------------- semantic process: vision feature --> semantic feature -------------------------------------------
        fusion_so = []

        for pair_idx, sub_rep, obj_rep, entity_embed in zip(rel_pair_idxs, sub_reps, obj_reps, entity_embeds):
            s_embed = self.W_sub(entity_embed[pair_idx[:, 0]])  #  Ws x ts
            o_embed = self.W_obj(entity_embed[pair_idx[:, 1]])  #  Wo x to

            sem_sub = self.vis2sem(sub_rep[pair_idx[:, 0]])  # h(xs)
            sem_obj = self.vis2sem(obj_rep[pair_idx[:, 1]])  # h(xo)
            
            gate_sem_sub = torch.sigmoid(self.gate_sub(cat((s_embed, sem_sub), dim=-1)))  # gs
            gate_sem_obj = torch.sigmoid(self.gate_obj(cat((o_embed, sem_obj), dim=-1)))  # go

            sub = s_embed + sem_sub * gate_sem_sub  # s = Ws x ts + gs · h(xs)  i.e., s = Ws x ts + vs
            obj = o_embed + sem_obj * gate_sem_obj  # o = Wo x to + go · h(xo)  i.e., o = Wo x to + vo

            ##### for the model convergence
            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)
            #####

            fusion_so.append(fusion_func(sub, obj)) # F(s, o)

        fusion_so = cat(fusion_so, dim=0)

        sem_pred = self.vis2sem(self.down_samp(union_features))  # h(xu)
        gate_sem_pred = torch.sigmoid(self.gate_pred(cat((fusion_so, sem_pred), dim=-1)))  # gp

        rel_rep = fusion_so - sem_pred * gate_sem_pred  #  F(s,o) - gp · h(xu)   i.e., r = F(s,o) - up
        predicate_proto = self.W_pred(self.rel_embed.weight)  # c = Wp x tp  i.e., semantic prototypes
        
        ##### for the model convergence
        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)

        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))
        ######

        # ------------------------------------------- semantic similarity -------------------------------------------
        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  #  <r_norm, c_norm> / τ
        # the rel_dists will be used to calculate the Le_sim with the ce_loss
        
        rel_dists=rel_dists+visual_rel_match
        # ************************************************************************************************************************************************************
        
        
        rel_dists = rel_dists.split(num_rels, dim=0)
        
        if self.training:
            add_losses.update(self.calculate_loss(predicate_proto,predicate_proto_norm,rel_labels,rel_rep))
        
        return entity_dists, rel_dists, add_losses, dict()
            
    def calculate_loss(self,predicate_proto,predicate_proto_norm,rel_labels,rel_rep):
        add_losses=dict()
        
        ### Prototype Regularization  ---- cosine similarity
        target_rpredicate_proto_norm = predicate_proto_norm.clone().detach() 
        simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
        l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (51*51)  
        add_losses.update({"l21_loss": l21})  # Le_sim = ||S||_{2,1}
        ### end
        
        ### Prototype Regularization  ---- Euclidean distance
        gamma2 = 7.0
        predicate_proto_a = predicate_proto.unsqueeze(dim=1).expand(-1, 51, -1) 
        predicate_proto_b = predicate_proto.detach().unsqueeze(dim=0).expand(51, -1, -1)
        proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
        sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
        topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1   # obtain d-, where k2 = 1
        dist_loss = torch.max(torch.zeros(51).cuda(), -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
        add_losses.update({"dist_loss2": dist_loss})
        ### end 

        ###  Prototype-based Learning  ---- Euclidean distance
        # rel_labels = cat(rel_labels, dim=0) if isinstance(rel_labels,(list,tuple)) else rel_labels
        gamma1 = 1.0
        rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, 51, -1)  # r
        predicate_proto_expand = predicate_proto.unsqueeze(dim=0).expand(rel_labels.size(0), -1, -1)  # ci
        distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2    # Distance Set G, gi = ||r-ci||_2^2
        mask_neg = torch.ones(rel_labels.size(0), 51).cuda()  
        mask_neg[torch.arange(rel_labels.size(0)), rel_labels] = 0
        distance_set_neg = distance_set * mask_neg
        distance_set_pos = distance_set[torch.arange(rel_labels.size(0)), rel_labels]  # gt i.e., g+
        sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
        topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10  # obtaining g-, where k1 = 10, 
        loss_sum = torch.max(torch.zeros(rel_labels.size(0)).cuda(), distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
        add_losses.update({"loss_dis": loss_sum})     # Le_euc = max(0, (g+) - (g-) + gamma1)
        ### end 
        
        return add_losses
    
    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        pos_embed = self.pos_embed(encode_box_info(proposals))

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        assert proposals[0].mode == 'xyxy'

        pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred)  # 512 -> 151
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs).long()
            else:
                obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()
        
        return obj_dists, obj_preds

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0 # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds

    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)  
        return x
    
    
def fusion_func(x, y):
    return F.relu(x + y) - (x - y) ** 2


class Message_Passing_Unit(nn.Module):
	def __init__(self, fea_size, filter_size = 128):
		super(Message_Passing_Unit, self).__init__()
		self.weight = nn.Linear(fea_size * 2, filter_size, bias=True) 
		self.fea_size = fea_size
		self.filter_size = filter_size

	def forward(self, unary_term, pair_term):

		if unary_term.size()[0] == 1 and pair_term.size()[0] > 1:
			unary_term = unary_term.expand(pair_term.size()[0], unary_term.size()[1])
		if unary_term.size()[0] > 1 and pair_term.size()[0] == 1:
			pair_term = pair_term.expand(unary_term.size()[0], pair_term.size()[1])
		
		gate = torch.cat([unary_term, pair_term], 1)
		gate = F.relu(gate)
		gate = F.sigmoid(self.weight(gate)).mean(1)

		output = pair_term * gate.view(-1, 1).expand(gate.size()[0], pair_term.size()[1])
		
		return output


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        ori_input = enc_input

        enc_input = self.norm_1(enc_input)
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)

        ori_input = enc_output + ori_input

        if non_pad_mask != None:
            enc_output *= non_pad_mask.float()

            enc_output = self.pos_ffn(self.norm_2(ori_input))
            enc_output *= non_pad_mask.float()
        else:
            enc_output = self.pos_ffn(self.norm_2(ori_input))

        enc_output = enc_output + ori_input
        return enc_output, enc_slf_attn


class EncoderCrossAttnLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderCrossAttnLayer, self).__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.slf_attn = nn.MultiheadAttention(d_model,n_head,dropout,kdim=d_k,vdim=d_v,batch_first=True)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)

    def forward(self, enc_input,k_input,v_input, non_pad_mask=None, slf_attn_mask=None):
        ori_input = enc_input

        enc_input = self.norm_q(enc_input)
        k_input = self.norm_k(k_input)
        v_input = self.norm_v(v_input)
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, k_input, v_input, attn_mask=slf_attn_mask)

        ori_input = enc_output + ori_input

        if non_pad_mask != None:
            enc_output *= non_pad_mask.float()

            enc_output = self.pos_ffn(self.norm_2(ori_input))
            enc_output *= non_pad_mask.float()
        else:
            enc_output = self.pos_ffn(self.norm_2(ori_input))

        enc_output = enc_output + ori_input
        return enc_output, enc_slf_attn
    
# def to_onehot(pre, target):
#     bil_target = torch.zeros(pre.shape, device=target.device).scatter_(1, torch.LongTensor(target.long().cpu()).to(
#         target.device), 1)
    
#     if not torch.all(bil_target.argmax(1) == target.squeeze()).item():
#         print('Warning: Target to one hot encoding failure .')
#         return None

#     return bil_target.to(target.device)