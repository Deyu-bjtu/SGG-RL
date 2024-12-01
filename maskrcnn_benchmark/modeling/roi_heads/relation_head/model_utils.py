import copy
import glob
import json
import maskrcnn_benchmark.config
import math
import os
import os.path
import random
import re
import time
import PIL
from PIL import Image
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from maskrcnn_benchmark.modeling.roi_heads.relation_head.attention_blocks import Attn_block,Trans_block
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_motifs import FrequencyBias
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_vctree import VCTreeLSTMContext
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_relation import layer_init
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.utils.comm import all_gather_with_grad, concat_all_gather, get_rank,find_linear_layers
from .utils_motifs import rel_vectors, obj_edge_vectors, to_onehot, nms_overlaps, encode_box_info 
from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.modeling.make_layers import make_fc
import transformers
import logging
from .attention_blocks import MLP,fusion_func
from transformers import BertConfig,BertModel,AutoTokenizer

class sec_branch(nn.Module):
    def __init__(self, config, in_channels):
        super(sec_branch, self).__init__()

        self.logger = logging.getLogger(__name__)
        embed_dim = config.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        roi_dim = config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        inner_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM

        num_head = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        dropout_rate = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        rel_layer = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER
        
        if config.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'
        self.config=config
        self.nms_thresh = config.TEST.RELATION.LATER_NMS_PREDICTION_THRES
        self.embed_dim=300
        
        statistics = get_dataset_statistics(config)

        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        rel_classes[rel_classes.index("__background__")]="background"
        obj_classes[obj_classes.index("__background__")]="background"
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        
        self.rel_prompt=[]
        for rel in rel_classes:
            self.rel_prompt.append(f'The relation word in this area is: {rel}')
        
        ##### refine object labels
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.config.GLOVE_DIR, wv_dim=self.embed_dim)  # load Glove for objects
        
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])
        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.obj_dim = in_channels
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_classes) 
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)

        
        # *********************************** init bert model ***********************************
        from transformers import BertTokenizer,BertModel
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.bert_encoder.pooler=None
        self.bert_cfg=self.bert_encoder.config
        
        from peft import LoraConfig,get_peft_model
        lora_target_modules = find_linear_layers(self.bert_encoder, ['query','value'])
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=lora_target_modules,
            bias="none",
        )
        self.bert_encoder = get_peft_model(self.bert_encoder, lora_config)
        self.bert_encoder.print_trainable_parameters()
        
        for n, p in self.bert_encoder.named_parameters():
            if 'embeddings' in n:
                self.logger.info(f"Calculate gradient name: {n}, param.shape: {p.shape}")
                p.requires_grad = True
        
        add_token_nums=self.add_token(rel_classes)
        self.init_tokenizer_weight(add_token_nums)
        
        self.img_cls=nn.Parameter(torch.randn(roi_dim))
        self.img_proj = nn.Sequential(
            nn.Linear(roi_dim, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.bert_cfg.hidden_size)
        )
        
        self.self_attention=nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(self.bert_cfg.hidden_size),
                nn.MultiheadAttention(self.bert_cfg.hidden_size, num_head,
                                      dropout_rate, batch_first=True),
                nn.LayerNorm(self.bert_cfg.hidden_size),
                MLP(self.bert_cfg.hidden_size,self.hidden_dim,self.bert_cfg.hidden_size,2),
            ]) for _ in range(rel_layer)
        ])
        
        self.cross_attention = nn.ModuleList([
            nn.ModuleList([
                # image-text cross transformer
                nn.LayerNorm(self.bert_cfg.hidden_size),
                nn.MultiheadAttention(self.bert_cfg.hidden_size, num_head,
                                      dropout_rate, batch_first=True),
                nn.LayerNorm(self.bert_cfg.hidden_size),
                MLP(self.bert_cfg.hidden_size,self.hidden_dim,self.bert_cfg.hidden_size,2),
                # text-image cross attention 
                nn.LayerNorm(self.bert_cfg.hidden_size),
                nn.MultiheadAttention(self.bert_cfg.hidden_size, num_head,
                                      dropout_rate, batch_first=True),
                nn.LayerNorm(self.bert_cfg.hidden_size),
                MLP(self.bert_cfg.hidden_size,self.hidden_dim,self.bert_cfg.hidden_size,2),
                
            ]) for _ in range(rel_layer)
        ])
        # image concate text 
        self.img_text_proj=MLP(self.bert_cfg.hidden_size,self.hidden_dim,self.bert_cfg.hidden_size,2)
        
        self.mask_to_rel=MLP(self.bert_cfg.hidden_size,self.hidden_dim,self.num_rel_cls,1)

    def add_token(self,rel_classes,external_tokens=[]):
        add_token_nums=0
        
        self.rel_map=dict()
        for rel_name in rel_classes:
            if len(self.tokenizer(rel_name).input_ids)>3:
                add_token_nums+=self.tokenizer.add_tokens(rel_name)

            token_ids=self.tokenizer(rel_name).input_ids
            assert len(token_ids)==3 or self.tokenizer.decode(token_ids[1])==rel_name
            self.rel_map[rel_name]=token_ids[1]
        
        self.all_label_token_ids=list(self.rel_map.values())
        self.logger.info(f'Relation map dict: {self.rel_map}')

        self.all_label_token_ids=torch.tensor(self.all_label_token_ids,dtype=torch.long,device=torch.device(f'cuda:{torch.cuda.current_device()}'))
        
        self.special_token_map=dict()
        for external_token in external_tokens:
            add_token_nums+= self.tokenizer.add_tokens(external_token)
            self.special_token_map[external_token]=self.tokenizer(external_token, add_special_tokens=False).input_ids[-1]  
            
        self.logger.info(f'Add token success, add token number: {add_token_nums}, external token id maps: {self.special_token_map}')
        
        return add_token_nums
    
    def init_tokenizer_weight(self,num_new_tokens):
        if num_new_tokens > 0:
            
            ori_input_embeddings=self.bert_encoder.get_input_embeddings().weight.data
            
            self.bert_encoder.resize_token_embeddings(len(self.tokenizer))
            
            input_embeddings = self.bert_encoder.get_input_embeddings().weight.data
            
            input_embeddings[:-num_new_tokens]=ori_input_embeddings
            
            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            
            input_embeddings[-num_new_tokens:] = input_embeddings_avg

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None,**kwargs):
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
        rel_tokenizer=self.tokenizer(self.rel_prompt, add_special_tokens=True, padding=True, return_tensors='pt').to(current_device)
        rel_encode_states=self.bert_encoder(**rel_tokenizer).last_hidden_state
        encode_rel_cls=rel_encode_states[:,0]
        
        if self.training:
            target_encode_rel_cls = encode_rel_cls.clone().detach() 
            simil_mat = encode_rel_cls @ target_encode_rel_cls.t()  # Semantic Matrix
            l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (51*51)  
            add_losses['l21_loss']=add_losses.get('l21_loss',0.0)+l21  # Le_sim = ||S||_{2,1}
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

            img_cls=self.img_cls.expand(batch_union_feature.shape[0],-1)
            align_vis=self.img_proj(torch.stack([img_cls,batch_union_feature,head_obj_feature,tail_obj_feature],dim=1)) # (num_rels,4,bert_dim)
            
            for (self_attn_ln,self_attn,self_mlp_ln,self_mlp) in self.self_attention:
                self_attn_vis,_=self_attn(align_vis,align_vis,align_vis)
                ln_self_attn_vis=self_attn_ln(self_attn_vis)+align_vis
                
                align_vis=self_mlp_ln(self_mlp(ln_self_attn_vis))+ln_self_attn_vis

            img_cls_pre=torch.matmul(align_vis[:,0,:],encode_rel_cls.permute(1,0).contiguous())
            
            if self.training:
                batch_rel_labels=rel_labels[batch_idx]
                gamma1 = 1.0
                rel_rep_expand = align_vis[:,0,:].unsqueeze(dim=1).expand(-1, 51, -1)  # r
                predicate_proto_expand = encode_rel_cls.unsqueeze(dim=0).expand(batch_rel_labels.size(0), -1, -1)  # ci
                distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2    # Distance Set G, gi = ||r-ci||_2^2
                mask_neg = torch.ones(batch_rel_labels.size(0), 51).cuda()  
                mask_neg[torch.arange(batch_rel_labels.size(0)), batch_rel_labels] = 0
                distance_set_neg = distance_set * mask_neg
                distance_set_pos = distance_set[torch.arange(batch_rel_labels.size(0)), batch_rel_labels]  # gt i.e., g+
                sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
                topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10  # obtaining g-, where k1 = 10, 
                loss_sum = torch.max(torch.zeros(batch_rel_labels.size(0)).cuda(), distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
                add_losses.update({"loss_dis": loss_sum})     # Le_euc = max(0, (g+) - (g-) + gamma1)
            # ********************************************* construct a relation prompt *********************************************
            rel_prompts = []
            for idx, (head_obj, tail_obj) in enumerate(zip(head_obj_pre, tail_obj_pre)):
                rel_prompt = f"Based on the above visual areas, the {self.obj_classes[head_obj]} is [MASK] the {self.obj_classes[tail_obj]}."
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
            
            # ************************************** relation mask prompt -- vision features attention ***********************************  
            for (it_attn_ln,it_attn,it_mlp_ln,it_mlp,ti_attn_ln,ti_attn,ti_mlp_ln,ti_mlp) in self.cross_attention:
                
                it_attn_vis,_=it_attn(align_vis,rel_prompt_embedding,rel_prompt_embedding)
                ti_attn_text,_=ti_attn(rel_prompt_embedding,align_vis,align_vis)
            
                ln_it_attn_vis,ln_ti_attn_text=it_attn_ln(it_attn_vis)+align_vis,ti_attn_ln(ti_attn_text)+rel_prompt_embedding
                
                it_mlp_vis,ti_mlp_text=it_mlp(ln_it_attn_vis),ti_mlp(ln_ti_attn_text)
                
                align_vis,rel_prompt_embedding=it_mlp_ln(it_mlp_vis)+ln_it_attn_vis,ti_mlp_ln(ti_mlp_text)+ln_ti_attn_text

            encoder_vis_text=self.bert_encoder.encoder(
                rel_prompt_embedding,
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
            mask_feature=sequence_output[mask_row,mask_col,:]
            
            mask_to_rel=self.mask_to_rel(mask_feature)
            bert_cls_sim=torch.matmul(bert_cls,encode_rel_cls.permute(1,0).contiguous())
            
            if self.training:
                add_losses['mask_to_rel']=add_losses.get('mask_to_rel',0.0)+F.cross_entropy(mask_to_rel,batch_rel_labels)
                add_losses['bert_cls_sim']=add_losses.get('bert_cls_sim',0.0)+F.cross_entropy(bert_cls_sim,batch_rel_labels)
                add_losses['img_cls_pre']=add_losses.get('img_cls_pre',0.0)+F.cross_entropy(img_cls_pre,batch_rel_labels)
            rel_dists.append(mask_to_rel+bert_cls_sim+img_cls_pre)
            
        return entity_dists, rel_dists, add_losses, dict()
    
    def prepare_bert_param(self,input_ids=None,inputs_embeds=None,past_key_values=None,encoder_hidden_states=None,token_type_ids=None,attention_mask=None,output_attentions=None,output_hidden_states=None,return_dict=None,head_mask=None):
        output_attentions = output_attentions if output_attentions is not None else self.bert_cfg.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.bert_cfg.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.bert_cfg.use_return_dict

        if self.bert_cfg.is_decoder:
            use_cache = use_cache if use_cache is not None else self.bert_cfg.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.bert_encoder.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
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
            if hasattr(self.bert_encoder.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.bert_encoder.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.bert_encoder.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.bert_cfg.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.bert_encoder.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.bert_encoder.get_head_mask(head_mask, self.bert_cfg.num_hidden_layers)
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
        use_gt_label = self.training or self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        pos_embed = self.pos_embed(encode_box_info(proposals))

        # label/logits embedding will be used as input
        if self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
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


class VLBERT(nn.Module):
    def __init__(self, config, in_channels):
        super(VLBERT, self).__init__()

        self.logger = logging.getLogger(__name__)
        embed_dim = config.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        roi_dim = config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        inner_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM

        num_head = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        dropout_rate = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        rel_layer = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER
        
        self.zeroshot_type=config.SOLVER.ZEROSHOT_MODE
        
        if config.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'
        self.config=config
        self.nms_thresh = config.TEST.RELATION.LATER_NMS_PREDICTION_THRES
        self.embed_dim=300
        
        statistics = get_dataset_statistics(config)
        
        obj_classes, rel_classes,fg_matrix = statistics['obj_classes'], statistics['rel_classes'],statistics['fg_matrix']
        rel_classes[rel_classes.index("__background__")]="background"
        obj_classes[obj_classes.index("__background__")]="background"
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
            
        ##### refine object labels
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.config.GLOVE_DIR, wv_dim=self.embed_dim)  # load Glove for objects
        
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])
        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.obj_dim = in_channels
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_classes) 
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)

        
        # *********************************** init bert model ***********************************
        from transformers import BertTokenizer,BertModel
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.bert_encoder.pooler=None
        self.bert_cfg=self.bert_encoder.config
        
        from peft import LoraConfig,get_peft_model
        lora_target_modules = find_linear_layers(self.bert_encoder, ['query','value'])
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=lora_target_modules,
            bias="none",
        )
        self.bert_encoder = get_peft_model(self.bert_encoder, lora_config)
        self.bert_encoder.print_trainable_parameters()
        
        for n, p in self.bert_encoder.named_parameters():
            if 'embeddings' in n:
                self.logger.info(f"Calculate gradient name: {n}, param.shape: {p.shape}")
                p.requires_grad = True
        
        
        add_token_nums=self.add_token(rel_classes+obj_classes)
        add_token_nums+=self.tokenizer.add_tokens(["[UNION]","[HEAD]","[TAIL]"])
        self.init_tokenizer_weight(self.bert_encoder,add_token_nums)
        
        self.rel_cls_score=nn.Parameter(torch.randn(self.bert_cfg.hidden_size))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.img_proj = nn.Sequential(
            nn.Linear(roi_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.bert_cfg.hidden_size)
        )
        
        self.head_gate=nn.Sequential(
            nn.Linear(2*self.bert_cfg.hidden_size,self.bert_cfg.hidden_size),
            nn.Sigmoid()
        )
        self.tail_gate=nn.Sequential(
            nn.Linear(2*self.bert_cfg.hidden_size,self.bert_cfg.hidden_size),
            nn.Sigmoid()
        )
        
        self.head_linear_fuse=nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.bert_cfg.hidden_size,self.bert_cfg.hidden_size),
                nn.ReLU(inplace=True),
            ),
            nn.LayerNorm(self.bert_cfg.hidden_size)
        ])
        self.tail_linear_fuse=nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.bert_cfg.hidden_size,self.bert_cfg.hidden_size),
                nn.ReLU(inplace=True),
            ),
            nn.LayerNorm(self.bert_cfg.hidden_size)
        ])
        self.union_linear_fuse=nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.bert_cfg.hidden_size,self.bert_cfg.hidden_size),
                nn.ReLU(inplace=True),
            ),
            nn.LayerNorm(self.bert_cfg.hidden_size)
        ])
        
        self.rel_gate=nn.Sequential(
            nn.Linear(2*self.bert_cfg.hidden_size,self.bert_cfg.hidden_size),
            nn.Sigmoid()
        )
        self.rel_linear_fuse=nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.bert_cfg.hidden_size,self.bert_cfg.hidden_size),
                nn.ReLU(inplace=True),
            ),
            nn.LayerNorm(self.bert_cfg.hidden_size)
        ])
        
        self.mask_to_rel=MLP(self.bert_cfg.hidden_size,self.hidden_dim,self.num_rel_cls,1)
        self.proj_pred=MLP(self.bert_cfg.hidden_size,self.hidden_dim,self.bert_cfg.hidden_size, 2)
        
        self.visual_fuse=nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(self.bert_cfg.hidden_size),
                nn.MultiheadAttention(self.bert_cfg.hidden_size, num_head,
                                      dropout_rate, batch_first=True),
                nn.LayerNorm(self.bert_cfg.hidden_size),
                MLP(self.bert_cfg.hidden_size,self.hidden_dim,self.bert_cfg.hidden_size,2),
            ]) for _ in range(rel_layer)
        ])
        self.rel_score=nn.Linear(self.bert_cfg.hidden_size,1)

        # self.freq_bias = FrequencyBias(config, statistics)
        # self.memory_bank=MemoryBank(50,self.bert_cfg.hidden_size,self.num_rel_cls,config.OUTPUT_DIR,device=torch.device(f'cuda:{torch.cuda.current_device()}'))

        # **************** loss ********************
        self.iter_num,self.gamma,self.total_iters=1,1,config.SOLVER.MAX_ITER
        bata=0.9999
        
        per_predicate_num=np.sum(fg_matrix.numpy(),axis=(0,1))
        self.per_predicate_weight=torch.tensor([(1-bata)/(1-bata**pre_num) for pre_num in per_predicate_num],dtype=torch.float)
        self.rel_ce_loss=nn.CrossEntropyLoss(self.per_predicate_weight)
              
    def add_token(self,rel_classes,external_tokens=[]):
        add_token_nums=0
        
        self.rel_map=dict()
        for rel_name in rel_classes:
            if len(self.tokenizer(rel_name).input_ids)>3:
                add_token_nums+=self.tokenizer.add_tokens(rel_name)

            token_ids=self.tokenizer(rel_name).input_ids
            assert len(token_ids)==3 or self.tokenizer.decode(token_ids[1])==rel_name
            self.rel_map[rel_name]=token_ids[1]
        
        self.all_label_token_ids=list(self.rel_map.values())
        self.logger.info(f'Relation map dict: {self.rel_map}')

        self.all_label_token_ids=torch.tensor(self.all_label_token_ids,dtype=torch.long,device=torch.device(f'cuda:{torch.cuda.current_device()}'))
        
        self.special_token_map=dict()
        for external_token in external_tokens:
            add_token_nums+= self.tokenizer.add_tokens(external_token)
            self.special_token_map[external_token]=self.tokenizer(external_token, add_special_tokens=False).input_ids[-1]  
            
        self.logger.info(f'Add token success, add token number: {add_token_nums}, external token id maps: {self.special_token_map}')
        
        return add_token_nums
    
    def init_tokenizer_weight(self,model,num_new_tokens):
        if num_new_tokens > 0:
            ori_input_embeddings=model.get_input_embeddings().weight.data
            
            model.resize_token_embeddings(len(self.tokenizer))
            
            input_embeddings = model.get_input_embeddings().weight.data
            input_embeddings[:-num_new_tokens]=ori_input_embeddings
            
            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            
            input_embeddings[-num_new_tokens:] = input_embeddings_avg

            self.logger.info(f'Update model embedding weight success')
    
    def calculate_loss(self,proposals,refine_logits,relation_logits,rel_labels):
        # ************************ relation loss ****************************
        relation_logits,rel_labels=torch.cat(relation_logits,dim=0),torch.cat(rel_labels,dim=0)
        rel_ce_loss=self.rel_ce_loss(relation_logits,rel_labels)
        
        rel_log_softmax = torch.log_softmax(relation_logits, dim=1)
        rel_logpt = torch.gather(rel_log_softmax, dim=1, index=rel_labels.view(-1, 1)).view(-1)
        
        rel_loss=max(1-(self.iter_num/self.total_iters),1e-7)*(1-torch.exp(rel_logpt))**self.gamma*rel_ce_loss
        rel_loss=torch.mean(rel_loss)  # torch.sum(f_loss)
        
        # **************************** object loss ***************************
        refine_obj_logits = cat(refine_logits, dim=0)
        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        
        obj_loss = F.cross_entropy(refine_obj_logits, fg_labels.long())
        
        # ********************************************************************
        
        self.iter_num+=1
        return rel_loss,obj_loss
      
    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None,**kwargs):
        current_device,add_losses,add_data=torch.device(f'cuda:{torch.cuda.current_device()}'),dict(),dict()
        
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
        encode_rel_cls=rel_encode_states[:,0,:]
        
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
                    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} maskrcnn_benchmark Warning: No Graph Detected ....')
                continue
            
            if self.training:
                batch_rel_labels=rel_labels[batch_idx]

            head_idx, tail_idx = batch_rel_pair_idx[:, 0], batch_rel_pair_idx[:, 1]
            head_obj_pre, tail_obj_pre = batch_obj_preds[head_idx], batch_obj_preds[tail_idx]
            head_obj_feature, tail_obj_feature = batch_roi_feature[head_idx], batch_roi_feature[tail_idx]

            align_roi_head,align_roi_tail,align_roi_union=self.img_proj(head_obj_feature),self.img_proj(tail_obj_feature),self.img_proj(batch_union_feature)

            # ************************************************ process union feature ************************************************
            union_fuse_obj=F.relu(align_roi_union+align_roi_head+align_roi_tail)-(align_roi_union-align_roi_head-align_roi_tail)**2
            union_fuse_obj=self.union_linear_fuse[1](union_fuse_obj+self.union_linear_fuse[0](union_fuse_obj))
            
            rel_cls_score=self.rel_cls_score.expand(union_fuse_obj.shape[0],-1)
            stack_roi_features=torch.stack([rel_cls_score,union_fuse_obj,align_roi_head,align_roi_tail],dim=1) # (num_rels,4,bert_dim)
            
            for (self_attn_ln,self_attn,self_mlp_ln,self_mlp) in self.visual_fuse:
                self_attn_vis,_=self_attn(stack_roi_features,stack_roi_features,stack_roi_features)
                ln_self_attn_vis=self_attn_ln(self_attn_vis)+stack_roi_features
                
                align_vis=self_mlp_ln(self_mlp(ln_self_attn_vis))+ln_self_attn_vis
            
            exist_rel_score=self.rel_score(align_vis[:,0,:])
                        
            add_data.setdefault('rel_scores',[]).append(exist_rel_score)

            # ********************************************* construct a relation prompt *********************************************
            rel_prompts,obj_prompts,gt_rel_prompts=[],[],[]
            for idx, (head_obj, tail_obj) in enumerate(zip(head_obj_pre, tail_obj_pre)):
                rel_prompt = f"Within this [UNION], the [HEAD] is [MASK] the [TAIL]"
                obj_prompt=f'{self.obj_classes[head_obj]} {self.obj_classes[tail_obj]}'
                if self.training:
                    gt_rel_prompt=f'The {self.obj_classes[head_obj]} is {self.rel_classes[batch_rel_labels[idx]]} the {self.obj_classes[tail_obj]}'
                    gt_rel_prompts.append(gt_rel_prompt)
                    
                rel_prompts.append(rel_prompt)
                obj_prompts.append(obj_prompt)   
            
            # ********************************************* encode visual-relation prompts *********************************************
            rel_prompt_tokenizer=self.tokenizer(rel_prompts,add_special_tokens=True,padding=True,return_tensors="pt").to(current_device)  # shape (num_rels,token_len,bert_dim) token[0]=[CLS]
            
            obj_prompt_tokenizer=self.tokenizer(obj_prompts,add_special_tokens=True,padding=True,return_tensors="pt").to(current_device) 
            assert obj_prompt_tokenizer.input_ids.shape[-1]==4,ValueError(obj_prompt_tokenizer.input_ids.shape)
            obj_prompt_embedding=self.bert_encoder.embeddings(
                        input_ids=obj_prompt_tokenizer.input_ids,
                        token_type_ids=obj_prompt_tokenizer.token_type_ids,
                        past_key_values_length=0)
            
            head_obj_embedding,tail_obj_embedding=obj_prompt_embedding[:,1,:],obj_prompt_embedding[:,2,:]  # object embedding weight
            head_gate,tail_gate=self.head_gate(torch.cat([head_obj_embedding,align_roi_head],dim=-1)),self.tail_gate(torch.cat([tail_obj_embedding,align_roi_tail],dim=-1))
            fused_sem_vis_head,fused_sem_vis_tail=align_roi_head*head_gate+head_obj_embedding,align_roi_tail*tail_gate+tail_obj_embedding
            fused_sem_vis_head,fused_sem_vis_tail=self.head_linear_fuse[1](fused_sem_vis_head+self.head_linear_fuse[0](fused_sem_vis_head)),self.tail_linear_fuse[1](fused_sem_vis_tail+self.tail_linear_fuse[0](fused_sem_vis_tail))
            
            mask_id=self.tokenizer('[MASK]',add_special_tokens=True, padding=True, return_tensors='pt').input_ids[0,1]
            mask_row,mask_col=torch.where(rel_prompt_tokenizer.input_ids==mask_id)
        
            extended_attention_mask,head_mask,encoder_hidden_states,encoder_extended_attention_mask,past_key_values,use_cache,output_attentions,output_hidden_states,return_dict,past_key_values_length=self.prepare_bert_param(**rel_prompt_tokenizer)
            rel_prompt_embedding=self.bert_encoder.embeddings(
                        input_ids=rel_prompt_tokenizer.input_ids,
                        token_type_ids=rel_prompt_tokenizer.token_type_ids,
                        past_key_values_length=past_key_values_length)
            
            union_token_ids,head_token_ids,tail_token_ids=self.tokenizer('[UNION]',add_special_tokens=True).input_ids,self.tokenizer('[HEAD]',add_special_tokens=True).input_ids,self.tokenizer('[TAIL]',add_special_tokens=True).input_ids
            assert len(union_token_ids)==3 and len(head_token_ids)==3 and len(tail_token_ids)==3

            union_token_id,head_token_id,tail_token_id=union_token_ids[1],head_token_ids[1],tail_token_ids[1]
            
            union_tokens=torch.where(rel_prompt_tokenizer.input_ids==union_token_id)
            head_tokens=torch.where(rel_prompt_tokenizer.input_ids==head_token_id)
            tail_tokens=torch.where(rel_prompt_tokenizer.input_ids==tail_token_id)
            
            rel_prompt_embedding[union_tokens]=rel_prompt_embedding[union_tokens]+union_fuse_obj
            rel_prompt_embedding[head_tokens]=rel_prompt_embedding[head_tokens]+fused_sem_vis_head
            rel_prompt_embedding[tail_tokens]=rel_prompt_embedding[tail_tokens]+fused_sem_vis_tail
            
            encoder_rel_text=self.bert_encoder.encoder(
                rel_prompt_embedding,
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
            rel_prompt_sequence_output=encoder_rel_text[0]
            rel_mask_feature=rel_prompt_sequence_output[mask_row,mask_col,:]
            
            mask_to_rel=self.mask_to_rel(rel_mask_feature)
            
            rel_mask_feature_norm=rel_mask_feature/rel_mask_feature.norm(dim=1,keepdim=True)
            encode_rel_cls_norm=encode_rel_cls/encode_rel_cls.norm(dim=1,keepdim=True)
            
            mask_rel_sim=rel_mask_feature_norm@encode_rel_cls_norm.t().contiguous()*self.logit_scale.exp()
            
            obj_roi_fused=F.relu(fused_sem_vis_head+fused_sem_vis_tail)-(fused_sem_vis_head-fused_sem_vis_tail)**2
            rel_gate_obj=self.rel_gate(torch.cat([obj_roi_fused,union_fuse_obj],dim=-1))
            visual_to_lg_rel_rep=obj_roi_fused+rel_gate_obj*union_fuse_obj
            
            visual_to_lg_rel_rep_norm=visual_to_lg_rel_rep/visual_to_lg_rel_rep.norm(dim=1,keepdim=True)
            rel_rep_cls=visual_to_lg_rel_rep_norm@encode_rel_cls_norm.t().contiguous()*self.logit_scale.exp()
            
            if self.training:
                gt_rel_prompt_tokenizer=self.tokenizer(gt_rel_prompts,add_special_tokens=True,padding=True,return_tensors="pt").to(current_device)  # shape (num_rels,token_len,bert_dim) token[0]=[CLS]
                encode_gt_rel_prompt_states=self.bert_encoder(**gt_rel_prompt_tokenizer)
                encode_gt_rel_prompt_cls=encode_gt_rel_prompt_states[0][:,0,:]
                
                union_rel_sem,gt_rel_prompt_sem=F.normalize(self.proj_pred(union_fuse_obj),dim=-1),F.normalize(self.proj_pred(encode_gt_rel_prompt_cls),dim=-1)
                confusion_matrix=union_rel_sem@gt_rel_prompt_sem.t().contiguous()
                
                pos_sim=1-torch.diag(confusion_matrix)
                neg_sim=confusion_matrix.clone()
                neg_sim.fill_diagonal_(0)
                vis_lg_sim = pos_sim.sum() + neg_sim.sum() / (confusion_matrix.shape[0] * (confusion_matrix.shape[0] - 1))  
                
                add_losses['vis_lg_sim']=add_losses.get('vis_lg_sim',0.0)+vis_lg_sim
                add_losses['vis_lg_ce']=add_losses.get('vis_lg_ce',0.0)+F.cross_entropy(confusion_matrix,torch.eye(confusion_matrix.shape[0],device=current_device))
                add_losses['mask_to_rel']=add_losses.get('mask_to_rel',0.0)+F.cross_entropy(mask_to_rel,batch_rel_labels)
                add_losses['mask_rel_sim']=add_losses.get('mask_rel_sim',0.0)+F.cross_entropy(mask_rel_sim,batch_rel_labels)
                
                exists_rel_label=copy.deepcopy(batch_rel_labels)
                exists_rel_label[exists_rel_label>0]=1
                add_losses['exist_rel']=add_losses.get('exist_rel',0.0)+F.binary_cross_entropy_with_logits(exist_rel_score.squeeze(-1),torch.tensor(exists_rel_label,device=current_device).float())

                extra_loss=self.calculate_semantic_loss(encode_rel_cls,encode_rel_cls_norm)
                extra_loss.update(self.calculate_similar_loss(encode_rel_cls,rel_mask_feature,batch_rel_labels))
                extra_loss.update(self.calculate_similar_loss(encode_rel_cls,visual_to_lg_rel_rep,batch_rel_labels,loss_name="visual_rel_dis"))
                
                # memory_loss=self.memory_bank(batch_rel_labels,visual_to_lg_rel_rep)
                
                # for key,value in memory_loss.items():
                #     add_losses[f'memory_{key}']=add_losses.get(f'memory_{key}',0.0)+value
                
                for key,value in extra_loss.items():
                    add_losses[key]=add_losses.get(key,0.0)+value
                
            # memory_pre=self.memory_bank.predict_similarity(rel_mask_feature)
            # rel_dists.append(rel_rep_cls+mask_rel_sim+memory_pre if memory_pre is not None else rel_rep_cls+mask_rel_sim)
            rel_dists.append(rel_rep_cls+mask_rel_sim)
            
        if self.training:
            add_data['final_loss']=dict()
            loss_relation,loss_refine=self.calculate_loss(proposals=proposals,refine_logits=entity_dists,relation_logits=rel_dists,rel_labels=rel_labels)
            add_data['final_loss']['loss_relation'],add_data['final_loss']['loss_refine']=loss_relation,loss_refine
        return entity_dists, rel_dists, add_losses, add_data
    
    def prepare_bert_param(self,input_ids=None,inputs_embeds=None,past_key_values=None,encoder_hidden_states=None,token_type_ids=None,attention_mask=None,output_attentions=None,output_hidden_states=None,return_dict=None,head_mask=None):
        output_attentions = output_attentions if output_attentions is not None else self.bert_cfg.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.bert_cfg.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.bert_cfg.use_return_dict

        if self.bert_cfg.is_decoder:
            use_cache = use_cache if use_cache is not None else self.bert_cfg.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.bert_encoder.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
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
            if hasattr(self.bert_encoder.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.bert_encoder.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.bert_encoder.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.bert_cfg.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.bert_encoder.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.bert_encoder.get_head_mask(head_mask, self.bert_cfg.num_hidden_layers)
        return extended_attention_mask,head_mask,encoder_hidden_states,encoder_extended_attention_mask,past_key_values,use_cache,output_attentions,output_hidden_states,return_dict,past_key_values_length
    
    def calculate_semantic_loss(self,semantic_feature,semantic_feature_norm):
        add_losses=dict()
        
        ### Prototype Regularization  ---- cosine similarity
        target_rpredicate_proto_norm = semantic_feature_norm.clone().detach() 
        simil_mat = semantic_feature_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
        l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (51*51)  
        add_losses.update({"l21_loss": l21})  # Le_sim = ||S||_{2,1}
        ### end
        
        ### Prototype Regularization  ---- Euclidean distance
        gamma2 = 7.0
        predicate_proto_a = semantic_feature.unsqueeze(dim=1).expand(-1, 51, -1) 
        predicate_proto_b = semantic_feature.detach().unsqueeze(dim=0).expand(51, -1, -1)
        proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
        sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
        topK_proto_dis = sorted_proto_dis_mat[:, :11].sum(dim=1) / 10   # obtain d-, where k2 = 1
        dist_loss = torch.max(torch.zeros(51).cuda(), -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
        add_losses.update({"dist_loss2": dist_loss})
        ### end
        
        return add_losses
        
    def calculate_similar_loss(self,semantic_feature,rel_rep,rel_labels,loss_name="loss_dis"):
        add_losses=dict()
        ###  Prototype-based Learning  ---- Euclidean distance
        # rel_labels = cat(rel_labels, dim=0) if isinstance(rel_labels,(list,tuple)) else rel_labels
        gamma1 = 1.0
        rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, semantic_feature.shape[0], -1)  # r
        predicate_proto_expand = semantic_feature.unsqueeze(dim=0).expand(rel_rep.size(0), -1, -1)  # ci
        distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2    # Distance Set G, gi = ||r-ci||_2^2
        mask_neg = torch.ones(rel_rep.size(0), semantic_feature.shape[0]).cuda()  
        mask_neg[torch.arange(rel_rep.size(0)), rel_labels] = 0
        distance_set_neg = distance_set * mask_neg
        distance_set_pos = distance_set[torch.arange(rel_rep.size(0)), rel_labels]  # gt i.e., g+
        sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
        topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10  # obtaining g-, where k1 = 10, 
        loss_sum = torch.max(torch.zeros(rel_rep.size(0)).cuda(), distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
        add_losses.update({loss_name: loss_sum})     # Le_euc = max(0, (g+) - (g-) + gamma1)
        ### end 
        
        return add_losses
    
    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        pos_embed = self.pos_embed(encode_box_info(proposals))

        # label/logits embedding will be used as input
        if self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
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


class EntityTrans(nn.Module):
    def __init__(self, config, in_channels):
        super(EntityTrans, self).__init__()

        self.logger = logging.getLogger(__name__)
        embed_dim = config.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        roi_dim = config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        inner_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM

        num_head = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        dropout_rate = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        rel_layer = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER
        
        if config.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'
        self.config=config
        self.nms_thresh = config.TEST.RELATION.LATER_NMS_PREDICTION_THRES
        
        statistics = get_dataset_statistics(config)
        
        obj_classes, rel_classes,fg_matrix = statistics['obj_classes'], statistics['rel_classes'],statistics['fg_matrix']
        self.num_obj_cls = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=config.GLOVE_DIR, wv_dim=embed_dim)  # load Glove for objects
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=embed_dim)   # load Glove for predicates
        self.obj_embed = nn.Embedding(self.num_obj_cls, embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
        
        ##### refine image/text features
        pretrain_clip_model='/data/sdb/pretrain_ckpt/CLIP/clip-vit-base-patch32'
        self.clip_processor=transformers.AutoProcessor.from_pretrained(pretrain_clip_model)
        self.clip_tokenizer=transformers.AutoTokenizer.from_pretrained(pretrain_clip_model)
        self.clip_vision_model=transformers.CLIPVisionModel.from_pretrained(pretrain_clip_model)

        self.align_img=make_fc(self.clip_vision_model.config.hidden_size,self.hidden_dim)
        
        ##### refine object labels
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])
        
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_cls) 
        self.lin_obj_cyx = make_fc(in_channels + embed_dim + 128, self.hidden_dim)

        ##### refine predicate spatial labels
        self.p_pos=make_fc(128,self.hidden_dim)
        self.p_entity=make_fc(in_channels,self.hidden_dim*2)
        
        self.rel_quary=nn.Parameter(torch.normal(mean=0, std=0.1, size=(self.hidden_dim,)))
        self.p_img_rep=nn.Sequential(
            nn.Conv2d(in_channels,self.hidden_dim,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=1,padding=0,stride=1)
        )
        
        self.rel_query_init=nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.Sequential(
                    nn.Linear(self.hidden_dim,inner_dim),
                    nn.ReLU(),
                    nn.Linear(inner_dim,self.hidden_dim),
                    nn.Dropout(dropout_rate)
                ),
                nn.LayerNorm(self.hidden_dim)
            ]) for _ in range(rel_layer)
        ])
        
        self.rel_query_refine=nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.Sequential(
                    nn.Linear(self.hidden_dim,inner_dim),
                    nn.ReLU(),
                    nn.Linear(inner_dim,self.hidden_dim),
                    nn.Dropout(dropout_rate)
                ),
                nn.LayerNorm(self.hidden_dim)
            ]) for _ in range(rel_layer)
        ])
        
        self.p_sub = MLP(embed_dim, self.hidden_dim // 2, self.hidden_dim, 2)
        self.p_obj = MLP(embed_dim, self.hidden_dim // 2, self.hidden_dim, 2)
        self.p_pred = MLP(embed_dim, self.hidden_dim // 2, self.hidden_dim, 2)

        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.hidden_dim, self.hidden_dim*2), nn.ReLU(True),
            nn.Dropout(dropout_rate), nn.Linear(self.hidden_dim*2, self.hidden_dim)
        ])
        
        self.gate_sub=make_fc(self.hidden_dim*2,self.hidden_dim)
        self.gate_obj=make_fc(self.hidden_dim*2,self.hidden_dim)
        self.gate_pred=make_fc(self.hidden_dim*2,self.hidden_dim)
        
        self.sample_union_rep=MLP(in_channels,self.hidden_dim,self.hidden_dim,2)
        
        self.filter_rel_rep=nn.Sequential(
            nn.Linear(self.hidden_dim,self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.filter_rel_norm=nn.LayerNorm(self.hidden_dim)
        self.drop_rel_rep=nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.fusion_triple_sem_rep=MLP(self.hidden_dim*3,self.hidden_dim,self.hidden_dim,2)
        self.refine_triple_rep=nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.Sequential(
                    nn.Linear(self.hidden_dim,inner_dim),
                    nn.ReLU(),
                    nn.Linear(inner_dim,self.hidden_dim),
                    nn.Dropout(dropout_rate)
                ),
                nn.LayerNorm(self.hidden_dim)
            ]) for _ in range(rel_layer)
        ])
        
        self.refine_union_vis=nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.Sequential(
                    nn.Linear(self.hidden_dim,inner_dim),
                    nn.ReLU(),
                    nn.Linear(inner_dim,self.hidden_dim),
                    nn.Dropout(dropout_rate)
                ),
                nn.LayerNorm(self.hidden_dim)
            ]) for _ in range(rel_layer)
        ])
        
        self.geo_rel_pre=nn.Linear(self.hidden_dim,self.num_rel_cls)
        self.sem_rel_pre=nn.Linear(self.hidden_dim,self.num_rel_cls)
        
        self.proj_head=MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim*2, 2)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # **************** loss ********************
        self.gamma,self.total_iters=1,config.SOLVER.MAX_ITER
        bata=0.9999
        
        per_predicate_num=np.sum(fg_matrix.numpy(),axis=(0,1))
        self.per_predicate_weight=torch.tensor([(1-bata)/(1-bata**pre_num) for pre_num in per_predicate_num],dtype=torch.float)
        self.rel_ce_loss=nn.CrossEntropyLoss(self.per_predicate_weight)

    def calculate_loss(self,proposals,refine_logits,relation_logits,rel_labels):
        # ************************ relation loss ****************************
        relation_logits,rel_labels=torch.cat(relation_logits,dim=0) if isinstance(relation_logits,(list,tuple)) else relation_logits,torch.cat(rel_labels,dim=0) if isinstance(rel_labels,(list,tuple)) else rel_labels
        rel_ce_loss=self.rel_ce_loss(relation_logits,rel_labels)
        
        rel_log_softmax = torch.log_softmax(relation_logits, dim=1)
        rel_logpt = torch.gather(rel_log_softmax, dim=1, index=rel_labels.view(-1, 1)).view(-1)
        
        rel_loss=(1-torch.exp(rel_logpt))**self.gamma*rel_ce_loss
        rel_loss=torch.mean(rel_loss)  # torch.sum(f_loss)
        
        # **************************** object loss ***************************
        refine_obj_logits = cat(refine_logits, dim=0) if isinstance(refine_logits,(list,tuple)) else refine_logits
        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        
        obj_loss = F.cross_entropy(refine_obj_logits, fg_labels.long())
        
        # ********************************************************************
        
        return rel_loss,obj_loss
      
    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None,**kwargs):
        current_device,add_losses,add_data=torch.device(f'cuda:{torch.cuda.current_device()}'),dict(),dict()
        
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)
        
        # refine object labels
        entity_dists, entity_preds, pos_embeds = self.refine_obj_labels(roi_features, proposals)
        ##### 

        entity_vis_rep=self.p_entity(roi_features)
        entity_vis_rep = entity_vis_rep.view(entity_vis_rep.size(0), 2, self.hidden_dim) # entity representation
        
        sub_vis_reps = entity_vis_rep[:, 1].contiguous().view(-1, self.hidden_dim).split(num_objs,dim=0)
        obj_vis_reps = entity_vis_rep[:, 0].contiguous().view(-1, self.hidden_dim).split(num_objs,dim=0)
        
        entity_dists = entity_dists.split(num_objs, dim=0)
        entity_sem_reps= self.obj_embed(entity_preds).split(num_objs,dim=0)
        pos_embeds=pos_embeds.split(num_objs,dim=0)
        # union_features=union_features.split(num_rels,dim=0)
        
        rel_sem_vector=self.p_pred(self.rel_embed.weight)
        
        rel_vis_reps,sub_sem_reps,obj_sem_reps,img_reps=[],[],[],[]
        for batch_idx,(proposal,sub_vis_rep,obj_vis_rep,entity_sem_rep,rel_pair_idx,pos_embed,union_feature) in enumerate(zip(proposals,sub_vis_reps,obj_vis_reps,entity_sem_reps,rel_pair_idxs,pos_embeds,union_features)):
            image = Image.open(proposal.get_field('file_name'))
            image_inputs = self.clip_processor(images=image, return_tensors="pt").to(current_device)
            img_encode_out=self.clip_vision_model(**image_inputs)
            img_rep = self.align_img(img_encode_out.last_hidden_state[:,1:,:])  # without cls token
            
            sub_pos_embed,obj_pos_embed=self.p_pos(pos_embed[rel_pair_idx[:,0]]),self.p_pos(pos_embed[rel_pair_idx[:,1]])
            sub_vis_rep,obj_vis_rep=sub_vis_rep[rel_pair_idx[:,0]],obj_vis_rep[rel_pair_idx[:,1]]
            sub_sem_rep,obj_sem_rep=entity_sem_rep[rel_pair_idx[:,0]],entity_sem_rep[rel_pair_idx[:,1]]
        
            # ********************************************* refine visual features ***************************************************
            sub_geo_rep,obj_geo_rep=sub_vis_rep+F.relu(sub_pos_embed),obj_vis_rep+F.relu(obj_pos_embed)
            rel_vis_rep,geo_vis_rep=self.rel_quary.expand(sub_geo_rep.shape[0],1,-1),torch.stack([sub_geo_rep,obj_geo_rep],dim=1)
            
            for (s_attn,s_norm,c_attn,c_norm,ffn,ffn_norm) in self.rel_query_init:
                attn_output, _ =s_attn(query=rel_vis_rep,key=rel_vis_rep,value=rel_vis_rep)
                rel_vis_rep=s_norm(rel_vis_rep+attn_output)
                
                attn_output, _ =c_attn(query=rel_vis_rep,key=geo_vis_rep,value=geo_vis_rep)
                rel_vis_rep=c_norm(rel_vis_rep+attn_output)
                
                rel_vis_rep=ffn_norm(ffn(rel_vis_rep)+rel_vis_rep)
            
            expand_img_rep=img_rep.expand(sub_geo_rep.shape[0],-1,-1)
            for (s_attn,s_norm,c_attn,c_norm,ffn,ffn_norm) in self.rel_query_refine:
                attn_output, _ =s_attn(query=rel_vis_rep,key=rel_vis_rep,value=rel_vis_rep)
                rel_vis_rep=s_norm(rel_vis_rep+attn_output)
                
                attn_output, _ =c_attn(query=rel_vis_rep,key=expand_img_rep,value=expand_img_rep)
                rel_vis_rep=c_norm(rel_vis_rep+attn_output)
                
                rel_vis_rep=ffn_norm(ffn(rel_vis_rep)+rel_vis_rep)

            rel_vis_rep=rel_vis_rep.squeeze()  # rel_num, hidden_dim
            rel_vis_reps.append(rel_vis_rep)
            
            # ********************************************* refine semantic features ***************************************************
            # refine object semantic features
            sub_sem_rep,obj_sem_rep=self.p_sub(sub_sem_rep),self.p_obj(obj_sem_rep)
            vis2sem_sub,vis2sem_obj,vis2sem_img=self.vis2sem(sub_vis_rep),self.vis2sem(obj_vis_rep),self.vis2sem(img_rep)
            
            gate_sub=F.sigmoid(self.gate_sub(torch.cat([sub_sem_rep,vis2sem_sub],dim=-1)))
            gate_obj=F.sigmoid(self.gate_obj(torch.cat([obj_sem_rep,vis2sem_obj],dim=-1)))
            
            sub_sem_rep,obj_sem_rep=sub_sem_rep+vis2sem_sub*gate_sub,obj_sem_rep+vis2sem_obj*gate_obj
            sub_sem_reps.append(sub_sem_rep)
            obj_sem_reps.append(obj_sem_rep)
            
            img_reps.append(vis2sem_img.expand(sub_geo_rep.shape[0],-1,-1))
            
        geo_rel_pre=self.geo_rel_pre(torch.cat(rel_vis_reps,dim=0))
        
        # refine predicate semantic features
        sub_sem_reps,obj_sem_reps=torch.cat(sub_sem_reps,dim=0),torch.cat(obj_sem_reps,dim=0)
        fusion_entity=F.relu(sub_sem_reps+obj_sem_reps)-(sub_sem_reps-obj_sem_reps)**2
        union_sem_reps=self.vis2sem(self.sample_union_rep(union_features))
        gate_union=F.sigmoid(self.gate_pred(torch.cat([fusion_entity,union_sem_reps],dim=-1)))
        
        rel_sem_reps=fusion_entity-union_sem_reps*gate_union
        rel_sem_reps=self.filter_rel_norm(self.filter_rel_rep(rel_sem_reps)+rel_sem_reps)
        rel_sem_reps=self.drop_rel_rep(rel_sem_reps)
        
        # refine triple semantic using image
        triple_sem_reps=torch.cat([sub_sem_reps,rel_sem_reps,obj_sem_reps],dim=-1)  
        triple_sem_reps=self.fusion_triple_sem_rep(triple_sem_reps).unsqueeze(1)
        img_reps=torch.cat(img_reps,dim=0)
        
        union_sem_reps=union_sem_reps.unsqueeze(1)
        for (s_attn,s_norm,ffn,ffn_norm) in self.refine_union_vis:
            attn_output, _ =s_attn(query=img_reps,key=union_sem_reps,value=union_sem_reps)
            img_reps=s_norm(img_reps+attn_output)
            
            img_reps=ffn_norm(ffn(img_reps)+img_reps)
        
        for (s_attn,s_norm,ffn,ffn_norm) in self.refine_triple_rep:
            attn_output, _ =s_attn(query=triple_sem_reps,key=img_reps,value=img_reps)
            triple_sem_reps=s_norm(triple_sem_reps+attn_output)
            
            triple_sem_reps=ffn_norm(ffn(triple_sem_reps)+triple_sem_reps)
        
        sem_rel_pre=self.sem_rel_pre(triple_sem_reps.squeeze())
        
        # semantic similarity
        rel_sem_vec=self.proj_head(self.drop_rel_rep(rel_sem_vector))
        rel_sem_reps=self.proj_head(rel_sem_reps)
        
        rel_sem_reps_norm = rel_sem_reps / rel_sem_reps.norm(dim=1, keepdim=True)  # r_norm
        rel_sem_vec_norm = rel_sem_vec / rel_sem_vec.norm(dim=1, keepdim=True)  # c_norm

        sem_rel_sim=rel_sem_reps_norm @ rel_sem_vec_norm.t() * self.logit_scale.exp()
        
        # final predicate dists
        rel_dists=geo_rel_pre+sem_rel_pre+sem_rel_sim
        
        if self.training:
            rel_labels=torch.cat(rel_labels,dim=0)
            
            obj_labels = [proposal.get_field("labels") for proposal in proposals]
            sub_embeds,obj_embeds=[],[]
            for rel_pair_idx,obj_label in zip(rel_pair_idxs,obj_labels):
                sub_objs,obj_objs=obj_label[rel_pair_idx[:,0]],obj_label[rel_pair_idx[:,1]]
                
                sub_embeds.append(self.p_sub(self.obj_embed(sub_objs.long())))
                obj_embeds.append(self.p_obj(self.obj_embed(obj_objs.long())))
            
            sub_embeds,obj_embeds,rel_embeds=torch.cat(sub_embeds,dim=0),torch.cat(obj_embeds,dim=0),self.p_pred(self.rel_embed(rel_labels))
            
            gt_triple_sem_reps=torch.cat([sub_embeds,rel_embeds,obj_embeds],dim=-1)  
            gt_triple_sem_reps=self.fusion_triple_sem_rep(gt_triple_sem_reps)
            # add_losses['triple_sem']=add_losses.get('triple_sem',0.0)+F.mse_loss(triple_sem_reps, gt_triple_sem_reps)
            triple_sim=F.cosine_similarity(triple_sem_reps.squeeze(),gt_triple_sem_reps,dim=1).sum()/triple_sem_reps.shape[0]
            add_losses['triple_sim']=add_losses.get('triple_sim',0.0)+(1-triple_sim)
            
            add_losses['geo_rel_pre']=add_losses.get('geo_rel_pre',0.0)+F.cross_entropy(geo_rel_pre,rel_labels)
            add_losses['sem_rel_pre']=add_losses.get('sem_rel_pre',0.0)+F.cross_entropy(sem_rel_pre,rel_labels)
            extra_loss=self.calculate_semantic_loss(rel_sem_vec,rel_sem_vec_norm)
            extra_loss.update(self.calculate_similar_loss(rel_sem_vec,rel_sem_reps,rel_labels))
            
            for key,value in extra_loss.items():
                add_losses[key]=add_losses.get(key,0.0)+value
                
            add_data['final_loss']=dict()
            loss_relation,loss_refine=self.calculate_loss(proposals=proposals,refine_logits=entity_dists,relation_logits=rel_dists,rel_labels=rel_labels)
            add_data['final_loss']['loss_relation'],add_data['final_loss']['loss_refine']=loss_relation,loss_refine
        
        rel_dists=rel_dists.split(num_rels,dim=0)
        return entity_dists, rel_dists, add_losses, add_data
    
    def calculate_semantic_loss(self,semantic_feature,semantic_feature_norm):
        add_losses=dict()
        
        ### Prototype Regularization  ---- cosine similarity
        target_rpredicate_proto_norm = semantic_feature_norm.clone().detach() 
        simil_mat = semantic_feature_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
        l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (51*51)  
        add_losses.update({"l21_loss": l21})  # Le_sim = ||S||_{2,1}
        ### end
        
        ### Prototype Regularization  ---- Euclidean distance
        gamma2 = 7.0
        predicate_proto_a = semantic_feature.unsqueeze(dim=1).expand(-1, 51, -1) 
        predicate_proto_b = semantic_feature.detach().unsqueeze(dim=0).expand(51, -1, -1)
        proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
        sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
        topK_proto_dis = sorted_proto_dis_mat[:, :11].sum(dim=1) / 10   # obtain d-, where k2 = 1
        dist_loss = torch.max(torch.zeros(51).cuda(), -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
        add_losses.update({"dist_loss2": dist_loss})
        ### end
        
        return add_losses
        
    def calculate_similar_loss(self,semantic_feature,rel_rep,rel_labels,loss_name="loss_dis"):
        add_losses=dict()
        ###  Prototype-based Learning  ---- Euclidean distance
        # rel_labels = cat(rel_labels, dim=0) if isinstance(rel_labels,(list,tuple)) else rel_labels
        gamma1 = 1.0
        rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, semantic_feature.shape[0], -1)  # r
        predicate_proto_expand = semantic_feature.unsqueeze(dim=0).expand(rel_rep.size(0), -1, -1)  # ci
        distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2    # Distance Set G, gi = ||r-ci||_2^2
        mask_neg = torch.ones(rel_rep.size(0), semantic_feature.shape[0]).cuda()  
        mask_neg[torch.arange(rel_rep.size(0)), rel_labels] = 0
        distance_set_neg = distance_set * mask_neg
        distance_set_pos = distance_set[torch.arange(rel_rep.size(0)), rel_labels]  # gt i.e., g+
        sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
        topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10  # obtaining g-, where k1 = 10, 
        loss_sum = torch.max(torch.zeros(rel_rep.size(0)).cuda(), distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
        add_losses.update({loss_name: loss_sum})     # Le_euc = max(0, (g+) - (g-) + gamma1)
        ### end 
        
        return add_losses
    
    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        pos_embed = self.pos_embed(encode_box_info(proposals))

        # label/logits embedding will be used as input
        if self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed.weight

        assert proposals[0].mode == 'xyxy'

        pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred)  # 512 -> 151
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs).long()
            else:
                obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()
        
        return obj_dists, obj_preds, pos_embed

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


class EntityTrans_v2(nn.Module):
    def __init__(self, config, in_channels):
        super(EntityTrans_v2, self).__init__()

        self.logger = logging.getLogger(__name__)
        embed_dim = config.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        roi_dim = config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        inner_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM

        num_head = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        dropout_rate = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        rel_layer = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER
        
        if config.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'
        self.config=config
        self.nms_thresh = config.TEST.RELATION.LATER_NMS_PREDICTION_THRES
        
        statistics = get_dataset_statistics(config)
        
        obj_classes, rel_classes,fg_matrix = statistics['obj_classes'], statistics['rel_classes'],statistics['fg_matrix']
        self.num_obj_cls = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=config.GLOVE_DIR, wv_dim=embed_dim)  # load Glove for objects
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=embed_dim)   # load Glove for predicates
        self.obj_embed = nn.Embedding(self.num_obj_cls, embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
        
        ##### refine image/text features
        pretrain_clip_model='/data/sdc/pretrain_model/CLIP/clip-vit-base-patch32'
        self.clip_processor=transformers.AutoProcessor.from_pretrained(pretrain_clip_model)
        self.clip_tokenizer=transformers.AutoTokenizer.from_pretrained(pretrain_clip_model)
        self.clip_vision_model=transformers.CLIPVisionModel.from_pretrained(pretrain_clip_model)

        self.align_img=make_fc(self.clip_vision_model.config.hidden_size,self.hidden_dim)
        
        ##### refine object labels
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])
        
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_cls) 
        self.lin_obj_cyx = make_fc(in_channels + embed_dim + 128, self.hidden_dim)

        ##### refine predicate spatial labels
        self.p_pos=make_fc(128,self.hidden_dim)
        self.p_entity=make_fc(in_channels,self.hidden_dim*2)
        
        self.rel_quary=nn.Parameter(torch.normal(mean=0, std=0.1, size=(self.hidden_dim,)))
        self.sem_rel_quary=nn.Parameter(torch.normal(mean=0, std=0.1, size=(self.hidden_dim,)))
        self.p_img_rep=nn.Sequential(
            nn.Conv2d(in_channels,self.hidden_dim,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=1,padding=0,stride=1)
        )
        
        self.rel_query_init=nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.Sequential(
                    nn.Linear(self.hidden_dim,inner_dim),
                    nn.ReLU(),
                    nn.Linear(inner_dim,self.hidden_dim),
                    nn.Dropout(dropout_rate)
                ),
                nn.LayerNorm(self.hidden_dim)
            ]) for _ in range(rel_layer)
        ])
        
        self.rel_query_refine=nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.Sequential(
                    nn.Linear(self.hidden_dim,inner_dim),
                    nn.ReLU(),
                    nn.Linear(inner_dim,self.hidden_dim),
                    nn.Dropout(dropout_rate)
                ),
                nn.LayerNorm(self.hidden_dim)
            ]) for _ in range(rel_layer)
        ])
        
        self.p_sub = MLP(embed_dim, self.hidden_dim // 2, self.hidden_dim, 2)
        self.p_obj = MLP(embed_dim, self.hidden_dim // 2, self.hidden_dim, 2)
        self.p_pred = MLP(embed_dim, self.hidden_dim // 2, self.hidden_dim, 2)

        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.hidden_dim, self.hidden_dim*2), nn.ReLU(True),
            nn.Dropout(dropout_rate), nn.Linear(self.hidden_dim*2, self.hidden_dim)
        ])
        
        self.gate_sub=make_fc(self.hidden_dim*2,self.hidden_dim)
        self.gate_obj=make_fc(self.hidden_dim*2,self.hidden_dim)
        self.gate_pred=make_fc(self.hidden_dim*2,self.hidden_dim)
        
        self.sample_union_rep=MLP(in_channels,self.hidden_dim,self.hidden_dim,2)
        
        self.filter_rel_rep=nn.Sequential(
            nn.Linear(self.hidden_dim,self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.filter_rel_norm=nn.LayerNorm(self.hidden_dim)
        self.drop_rel_rep=nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.refine_union_vis=nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.Sequential(
                    nn.Linear(self.hidden_dim,inner_dim),
                    nn.ReLU(),
                    nn.Linear(inner_dim,self.hidden_dim),
                    nn.Dropout(dropout_rate)
                ),
                nn.LayerNorm(self.hidden_dim)
            ]) for _ in range(rel_layer)
        ])
        
        self.init_sem_rel_query=nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.Sequential(
                    nn.Linear(self.hidden_dim,inner_dim),
                    nn.ReLU(),
                    nn.Linear(inner_dim,self.hidden_dim),
                    nn.Dropout(dropout_rate)
                ),
                nn.LayerNorm(self.hidden_dim)
            ]) for _ in range(rel_layer)
        ])
        self.refine_sem_rel_query=nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.Sequential(
                    nn.Linear(self.hidden_dim,inner_dim),
                    nn.ReLU(),
                    nn.Linear(inner_dim,self.hidden_dim),
                    nn.Dropout(dropout_rate)
                ),
                nn.LayerNorm(self.hidden_dim)
            ]) for _ in range(rel_layer)
        ])
        
        
        self.geo_rel_pre=nn.Linear(self.hidden_dim,self.num_rel_cls)
        self.sem_rel_pre=nn.Linear(self.hidden_dim,self.num_rel_cls)
        
        self.proj_head=MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim*2, 2)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # **************** loss ********************
        self.gamma,self.total_iters=1,config.SOLVER.MAX_ITER
        bata=0.9999
        
        per_predicate_num=np.sum(fg_matrix.numpy(),axis=(0,1))
        self.per_predicate_weight=torch.tensor([(1-bata)/(1-bata**pre_num) for pre_num in per_predicate_num],dtype=torch.float)
        self.rel_ce_loss=nn.CrossEntropyLoss(self.per_predicate_weight)

    def calculate_loss(self,proposals,refine_logits,relation_logits,rel_labels):
        # ************************ relation loss ****************************
        relation_logits,rel_labels=torch.cat(relation_logits,dim=0) if isinstance(relation_logits,(list,tuple)) else relation_logits,torch.cat(rel_labels,dim=0) if isinstance(rel_labels,(list,tuple)) else rel_labels
        rel_ce_loss=self.rel_ce_loss(relation_logits,rel_labels)
        
        rel_log_softmax = torch.log_softmax(relation_logits, dim=1)
        rel_logpt = torch.gather(rel_log_softmax, dim=1, index=rel_labels.view(-1, 1)).view(-1)
        
        rel_loss=(1-torch.exp(rel_logpt))**self.gamma*rel_ce_loss
        rel_loss=torch.mean(rel_loss)  # torch.sum(f_loss)
        
        # **************************** object loss ***************************
        refine_obj_logits = cat(refine_logits, dim=0) if isinstance(refine_logits,(list,tuple)) else refine_logits
        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        
        obj_loss = F.cross_entropy(refine_obj_logits, fg_labels.long())
        
        # ********************************************************************
        
        return rel_loss,obj_loss
      
    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None,**kwargs):
        current_device,add_losses,add_data=torch.device(f'cuda:{torch.cuda.current_device()}'),dict(),dict()
        
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)
        
        # refine object labels
        entity_dists, entity_preds, pos_embeds = self.refine_obj_labels(roi_features, proposals)
        ##### 

        entity_vis_rep=self.p_entity(roi_features)
        entity_vis_rep = entity_vis_rep.view(entity_vis_rep.size(0), 2, self.hidden_dim) # entity representation
        
        sub_vis_reps = entity_vis_rep[:, 1].contiguous().view(-1, self.hidden_dim).split(num_objs,dim=0)
        obj_vis_reps = entity_vis_rep[:, 0].contiguous().view(-1, self.hidden_dim).split(num_objs,dim=0)
        
        entity_dists = entity_dists.split(num_objs, dim=0)
        entity_sem_reps= self.obj_embed(entity_preds).split(num_objs,dim=0)
        pos_embeds=pos_embeds.split(num_objs,dim=0)
        # union_features=union_features.split(num_rels,dim=0)
        
        rel_sem_vector=self.p_pred(self.rel_embed.weight)
        
        rel_vis_reps,sub_sem_reps,obj_sem_reps,img_reps=[],[],[],[]
        for batch_idx,(proposal,sub_vis_rep,obj_vis_rep,entity_sem_rep,rel_pair_idx,pos_embed,union_feature) in enumerate(zip(proposals,sub_vis_reps,obj_vis_reps,entity_sem_reps,rel_pair_idxs,pos_embeds,union_features)):
            image = Image.open(proposal.get_field('file_name'))
            image_inputs = self.clip_processor(images=image, return_tensors="pt").to(current_device)
            img_encode_out=self.clip_vision_model(**image_inputs)
            img_rep = self.align_img(img_encode_out.last_hidden_state[:,1:,:])  # without cls token
            
            sub_pos_embed,obj_pos_embed=self.p_pos(pos_embed[rel_pair_idx[:,0]]),self.p_pos(pos_embed[rel_pair_idx[:,1]])
            sub_vis_rep,obj_vis_rep=sub_vis_rep[rel_pair_idx[:,0]],obj_vis_rep[rel_pair_idx[:,1]]
            sub_sem_rep,obj_sem_rep=entity_sem_rep[rel_pair_idx[:,0]],entity_sem_rep[rel_pair_idx[:,1]]
        
            # ********************************************* refine visual features ***************************************************
            sub_geo_rep,obj_geo_rep=sub_vis_rep+F.relu(sub_pos_embed),obj_vis_rep+F.relu(obj_pos_embed)
            rel_vis_rep,geo_vis_rep=self.rel_quary.expand(sub_geo_rep.shape[0],1,-1),torch.stack([sub_geo_rep,obj_geo_rep],dim=1)
            
            for (s_attn,s_norm,c_attn,c_norm,ffn,ffn_norm) in self.rel_query_init:
                attn_output, _ =s_attn(query=rel_vis_rep,key=rel_vis_rep,value=rel_vis_rep)
                rel_vis_rep=s_norm(rel_vis_rep+attn_output)
                
                attn_output, _ =c_attn(query=rel_vis_rep,key=geo_vis_rep,value=geo_vis_rep)
                rel_vis_rep=c_norm(rel_vis_rep+attn_output)
                
                rel_vis_rep=ffn_norm(ffn(rel_vis_rep)+rel_vis_rep)
            
            expand_img_rep=img_rep.expand(sub_geo_rep.shape[0],-1,-1)
            for (s_attn,s_norm,c_attn,c_norm,ffn,ffn_norm) in self.rel_query_refine:
                attn_output, _ =s_attn(query=rel_vis_rep,key=rel_vis_rep,value=rel_vis_rep)
                rel_vis_rep=s_norm(rel_vis_rep+attn_output)
                
                attn_output, _ =c_attn(query=rel_vis_rep,key=expand_img_rep,value=expand_img_rep)
                rel_vis_rep=c_norm(rel_vis_rep+attn_output)
                
                rel_vis_rep=ffn_norm(ffn(rel_vis_rep)+rel_vis_rep)

            rel_vis_rep=rel_vis_rep.squeeze()  # rel_num, hidden_dim
            rel_vis_reps.append(rel_vis_rep)
            
            # ********************************************* refine semantic features ***************************************************
            # refine object semantic features
            sub_sem_rep,obj_sem_rep=self.p_sub(sub_sem_rep),self.p_obj(obj_sem_rep)
            vis2sem_sub,vis2sem_obj,vis2sem_img=self.vis2sem(sub_vis_rep),self.vis2sem(obj_vis_rep),self.vis2sem(img_rep)
            
            gate_sub=F.sigmoid(self.gate_sub(torch.cat([sub_sem_rep,vis2sem_sub],dim=-1)))
            gate_obj=F.sigmoid(self.gate_obj(torch.cat([obj_sem_rep,vis2sem_obj],dim=-1)))
            
            sub_sem_rep,obj_sem_rep=sub_sem_rep+vis2sem_sub*gate_sub,obj_sem_rep+vis2sem_obj*gate_obj
            sub_sem_reps.append(sub_sem_rep)
            obj_sem_reps.append(obj_sem_rep)
            
            img_reps.append(vis2sem_img.expand(sub_geo_rep.shape[0],-1,-1))
            
        geo_rel_pre=self.geo_rel_pre(torch.cat(rel_vis_reps,dim=0))
        
        # refine predicate semantic features
        sub_sem_reps,obj_sem_reps=torch.cat(sub_sem_reps,dim=0),torch.cat(obj_sem_reps,dim=0)
        fusion_entity=F.relu(sub_sem_reps+obj_sem_reps)-(sub_sem_reps-obj_sem_reps)**2
        union_sem_reps=self.vis2sem(self.sample_union_rep(union_features))
        gate_union=F.sigmoid(self.gate_pred(torch.cat([fusion_entity,union_sem_reps],dim=-1)))
        
        rel_sem_reps=fusion_entity-union_sem_reps*gate_union
        rel_sem_reps=self.filter_rel_norm(self.filter_rel_rep(rel_sem_reps)+rel_sem_reps)
        rel_sem_reps=self.drop_rel_rep(rel_sem_reps)
        
        # refine triple semantic using image
        triple_sem_reps=torch.stack([sub_sem_reps,rel_sem_reps,obj_sem_reps],dim=1)
        img_reps=torch.cat(img_reps,dim=0)
        
        union_sem_reps,sem_rel_query=union_sem_reps.unsqueeze(1),self.sem_rel_quary.expand(img_reps.shape[0],1,-1)
        for (s_attn,s_norm,ffn,ffn_norm) in self.refine_union_vis:
            attn_output, _ =s_attn(query=img_reps,key=union_sem_reps,value=union_sem_reps)
            img_reps=s_norm(img_reps+attn_output)
            
            img_reps=ffn_norm(ffn(img_reps)+img_reps)
        
        for (s_attn,s_norm,c_attn,c_norm,ffn,ffn_norm) in self.init_sem_rel_query:
            s_attn_output, _= s_attn(query=sem_rel_query,key=sem_rel_query,value=sem_rel_query)
            sem_rel_query=s_norm(sem_rel_query+s_attn_output)
            
            attn_output, _ =c_attn(query=sem_rel_query,key=triple_sem_reps,value=triple_sem_reps)
            sem_rel_query=c_norm(sem_rel_query+attn_output)
            
            sem_rel_query=ffn_norm(ffn(sem_rel_query)+sem_rel_query)
        
        for (s_attn,s_norm,c_attn,c_norm,ffn,ffn_norm) in self.refine_sem_rel_query:
            s_attn_output, _= s_attn(query=sem_rel_query,key=sem_rel_query,value=sem_rel_query)
            sem_rel_query=s_norm(sem_rel_query+s_attn_output)
            
            attn_output, _ =c_attn(query=sem_rel_query,key=img_reps,value=img_reps)
            sem_rel_query=c_norm(sem_rel_query+attn_output)
            
            sem_rel_query=ffn_norm(ffn(sem_rel_query)+sem_rel_query)
        
        sem_rel_pre=self.sem_rel_pre(sem_rel_query.squeeze())
        
        # semantic similarity
        rel_sem_vec=self.proj_head(self.drop_rel_rep(rel_sem_vector))
        rel_sem_reps=self.proj_head(rel_sem_reps)
        
        rel_sem_reps_norm = rel_sem_reps / rel_sem_reps.norm(dim=1, keepdim=True)  # r_norm
        rel_sem_vec_norm = rel_sem_vec / rel_sem_vec.norm(dim=1, keepdim=True)  # c_norm

        sem_rel_sim=rel_sem_reps_norm @ rel_sem_vec_norm.t() * self.logit_scale.exp()
        
        # final predicate dists
        rel_dists=geo_rel_pre+sem_rel_pre+sem_rel_sim
        
        if self.training:
            rel_labels=torch.cat(rel_labels,dim=0)
            
            rel_embeds=self.p_pred(self.rel_embed(rel_labels))
            rel_sim=F.cosine_similarity(sem_rel_query.squeeze(),rel_embeds,dim=1).sum()/sem_rel_query.shape[0]
            add_losses['rel_sim']=add_losses.get('rel_sim',0.0)+(1-rel_sim)
            
            add_losses['geo_rel_pre']=add_losses.get('geo_rel_pre',0.0)+F.cross_entropy(geo_rel_pre,rel_labels)
            add_losses['sem_rel_pre']=add_losses.get('sem_rel_pre',0.0)+F.cross_entropy(sem_rel_pre,rel_labels)
            extra_loss=self.calculate_semantic_loss(rel_sem_vec,rel_sem_vec_norm)
            extra_loss.update(self.calculate_similar_loss(rel_sem_vec,rel_sem_reps,rel_labels))
            
            for key,value in extra_loss.items():
                add_losses[key]=add_losses.get(key,0.0)+value
                
            add_data['final_loss']=dict()
            loss_relation,loss_refine=self.calculate_loss(proposals=proposals,refine_logits=entity_dists,relation_logits=rel_dists,rel_labels=rel_labels)
            add_data['final_loss']['loss_relation'],add_data['final_loss']['loss_refine']=loss_relation,loss_refine
        
        rel_dists=rel_dists.split(num_rels,dim=0)
        return entity_dists, rel_dists, add_losses, add_data
    
    def calculate_semantic_loss(self,semantic_feature,semantic_feature_norm):
        add_losses=dict()
        
        ### Prototype Regularization  ---- cosine similarity
        target_rpredicate_proto_norm = semantic_feature_norm.clone().detach() 
        simil_mat = semantic_feature_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
        l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (51*51)  
        add_losses.update({"l21_loss": l21})  # Le_sim = ||S||_{2,1}
        ### end
        
        ### Prototype Regularization  ---- Euclidean distance
        gamma2 = 7.0
        predicate_proto_a = semantic_feature.unsqueeze(dim=1).expand(-1, 51, -1) 
        predicate_proto_b = semantic_feature.detach().unsqueeze(dim=0).expand(51, -1, -1)
        proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
        sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
        topK_proto_dis = sorted_proto_dis_mat[:, :11].sum(dim=1) / 10   # obtain d-, where k2 = 1
        dist_loss = torch.max(torch.zeros(51).cuda(), -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
        add_losses.update({"dist_loss2": dist_loss})
        ### end
        
        return add_losses
        
    def calculate_similar_loss(self,semantic_feature,rel_rep,rel_labels,loss_name="loss_dis"):
        add_losses=dict()
        ###  Prototype-based Learning  ---- Euclidean distance
        # rel_labels = cat(rel_labels, dim=0) if isinstance(rel_labels,(list,tuple)) else rel_labels
        gamma1 = 1.0
        rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, semantic_feature.shape[0], -1)  # r
        predicate_proto_expand = semantic_feature.unsqueeze(dim=0).expand(rel_rep.size(0), -1, -1)  # ci
        distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2    # Distance Set G, gi = ||r-ci||_2^2
        mask_neg = torch.ones(rel_rep.size(0), semantic_feature.shape[0]).cuda()  
        mask_neg[torch.arange(rel_rep.size(0)), rel_labels] = 0
        distance_set_neg = distance_set * mask_neg
        distance_set_pos = distance_set[torch.arange(rel_rep.size(0)), rel_labels]  # gt i.e., g+
        sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
        topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10  # obtaining g-, where k1 = 10, 
        loss_sum = torch.max(torch.zeros(rel_rep.size(0)).cuda(), distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
        add_losses.update({loss_name: loss_sum})     # Le_euc = max(0, (g+) - (g-) + gamma1)
        ### end 
        
        return add_losses
    
    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        pos_embed = self.pos_embed(encode_box_info(proposals))

        # label/logits embedding will be used as input
        if self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed.weight

        assert proposals[0].mode == 'xyxy'

        pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred)  # 512 -> 151
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs).long()
            else:
                obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()
        
        return obj_dists, obj_preds, pos_embed

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


class EntityTrans_v3(nn.Module):
    def __init__(self, config, in_channels):
        super(EntityTrans_v3, self).__init__()

        self.logger = logging.getLogger(__name__)
        embed_dim = config.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        roi_dim = config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        inner_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM

        num_head = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        dropout_rate = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        rel_layer = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER
        
        if config.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.config=config
        self.nms_thresh = config.TEST.RELATION.LATER_NMS_PREDICTION_THRES
        
        statistics = get_dataset_statistics(config)
        
        obj_classes, rel_classes,fg_matrix = statistics['obj_classes'], statistics['rel_classes'],statistics['fg_matrix']
        self.num_obj_cls = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=config.GLOVE_DIR, wv_dim=embed_dim)  # load Glove for objects
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=embed_dim)   # load Glove for predicates
        self.obj_embed = nn.Embedding(self.num_obj_cls, embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
        
        ##### refine image/text features
        pretrain_clip_model='/data/sdb/pretrain_ckpt/CLIP/clip-vit-base-patch32'
        self.clip_processor=transformers.AutoProcessor.from_pretrained(pretrain_clip_model)
        self.clip_tokenizer=transformers.AutoTokenizer.from_pretrained(pretrain_clip_model)
        self.clip_vision_model=transformers.CLIPVisionModel.from_pretrained(pretrain_clip_model)

        self.align_img=make_fc(self.clip_vision_model.config.hidden_size,self.hidden_dim)
                
        if self.mode == 'predcls' or self.mode=='sgdet':
            ##### refine object labels
            self.pos_embed = nn.Sequential(*[
                nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
                nn.Linear(32, 128), nn.ReLU(inplace=True),
            ])
            
            self.out_obj = make_fc(self.hidden_dim, self.num_obj_cls) 
            self.lin_obj_cyx = make_fc(in_channels + embed_dim + 128, self.hidden_dim)
            self.p_pos=make_fc(128,self.hidden_dim)
            self.p_entity=make_fc(in_channels,self.hidden_dim*2)
        elif self.mode=='sgcls':
            # init contextual lstm encoding
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        else:
            raise ValueError(f'Unknow mode: {self.mode}')
            
        self.rel_quary=nn.Parameter(torch.normal(mean=0, std=0.1, size=(self.hidden_dim,)))
        self.sem_rel_quary=nn.Parameter(torch.normal(mean=0, std=0.1, size=(self.hidden_dim,)))
        self.p_img_rep=nn.Sequential(
            nn.Conv2d(in_channels,self.hidden_dim,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=1,padding=0,stride=1)
        )
        
        self.rel_query_init=nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.Sequential(
                    nn.Linear(self.hidden_dim,inner_dim),
                    nn.ReLU(),
                    nn.Linear(inner_dim,self.hidden_dim),
                    nn.Dropout(dropout_rate)
                ),
                nn.LayerNorm(self.hidden_dim)
            ]) for _ in range(rel_layer)
        ])
        
        self.rel_query_refine=nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.Sequential(
                    nn.Linear(self.hidden_dim,inner_dim),
                    nn.ReLU(),
                    nn.Linear(inner_dim,self.hidden_dim),
                    nn.Dropout(dropout_rate)
                ),
                nn.LayerNorm(self.hidden_dim)
            ]) for _ in range(rel_layer)
        ])
        self.p_sub = MLP(embed_dim, self.hidden_dim // 2, self.hidden_dim, 2)
        self.p_obj = MLP(embed_dim, self.hidden_dim // 2, self.hidden_dim, 2)
        self.p_pred = MLP(embed_dim, self.hidden_dim // 2, self.hidden_dim, 2)
        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.hidden_dim, self.hidden_dim*2), nn.ReLU(True),
            nn.Dropout(dropout_rate), nn.Linear(self.hidden_dim*2, self.hidden_dim)
        ])
        
        self.gate_sub=make_fc(self.hidden_dim*2,self.hidden_dim)
        self.gate_obj=make_fc(self.hidden_dim*2,self.hidden_dim)
        self.gate_pred=make_fc(self.hidden_dim*2,self.hidden_dim)
        
        self.sample_union_rep=MLP(in_channels,self.hidden_dim,self.hidden_dim,2)
        
        self.filter_rel_rep=nn.Sequential(
            nn.Linear(self.hidden_dim,self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.filter_rel_norm=nn.LayerNorm(self.hidden_dim)
        self.drop_rel_rep=nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.refine_union_vis=nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.Sequential(
                    nn.Linear(self.hidden_dim,inner_dim),
                    nn.ReLU(),
                    nn.Linear(inner_dim,self.hidden_dim),
                    nn.Dropout(dropout_rate)
                ),
                nn.LayerNorm(self.hidden_dim)
            ]) for _ in range(rel_layer)
        ])
        
        self.init_sem_rel_query=nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.Sequential(
                    nn.Linear(self.hidden_dim,inner_dim),
                    nn.ReLU(),
                    nn.Linear(inner_dim,self.hidden_dim),
                    nn.Dropout(dropout_rate)
                ),
                nn.LayerNorm(self.hidden_dim)
            ]) for _ in range(rel_layer)
        ])
        self.refine_sem_rel_query=nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.Sequential(
                    nn.Linear(self.hidden_dim,inner_dim),
                    nn.ReLU(),
                    nn.Linear(inner_dim,self.hidden_dim),
                    nn.Dropout(dropout_rate)
                ),
                nn.LayerNorm(self.hidden_dim)
            ]) for _ in range(rel_layer)
        ])
        self.geo_rel_pre=nn.Linear(self.hidden_dim,self.num_rel_cls)
        self.sem_rel_pre=nn.Linear(self.hidden_dim,self.num_rel_cls)
        
        self.fusion_triple_sem_rep=MLP(self.hidden_dim*3,self.hidden_dim,self.hidden_dim,2)
        self.triple_sem_pre=nn.Linear(self.hidden_dim,self.num_rel_cls)
        
        self.proj_head=MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim*2, 2)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # **************** loss ********************
        self.gamma,self.total_iters=1,config.SOLVER.MAX_ITER
        bata=0.9999
        
        per_predicate_num=np.sum(fg_matrix.numpy(),axis=(0,1))
        self.per_predicate_weight=torch.tensor([(1-bata)/(1-bata**pre_num) for pre_num in per_predicate_num],dtype=torch.float)
        self.rel_ce_loss=nn.CrossEntropyLoss(self.per_predicate_weight)
        
        # **************** predicate prediction weights ********************
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)
        if self.mode=='sgcls':
            self.geo_rel_w,self.sem_rel_w,self.sem_rel_sim_w,self.triple_sem_rel_w=nn.Parameter(torch.ones((self.num_rel_cls,))),nn.Parameter(torch.ones((self.num_rel_cls,))),nn.Parameter(torch.ones((self.num_rel_cls,))),nn.Parameter(torch.ones((self.num_rel_cls,)))
            self.freq_weights=nn.Parameter(torch.ones((self.num_rel_cls,)))
            
    def calculate_loss(self,proposals,refine_logits,relation_logits,rel_labels):
        # ************************ relation loss ****************************
        relation_logits,rel_labels=torch.cat(relation_logits,dim=0) if isinstance(relation_logits,(list,tuple)) else relation_logits,torch.cat(rel_labels,dim=0) if isinstance(rel_labels,(list,tuple)) else rel_labels
        rel_ce_loss=self.rel_ce_loss(relation_logits,rel_labels)
        
        rel_log_softmax = torch.log_softmax(relation_logits, dim=1)
        rel_logpt = torch.gather(rel_log_softmax, dim=1, index=rel_labels.view(-1, 1)).view(-1)
        
        rel_loss=(1-torch.exp(rel_logpt))**self.gamma*rel_ce_loss
        rel_loss=torch.mean(rel_loss)  # torch.sum(f_loss)
        
        # **************************** object loss ***************************
        refine_obj_logits = cat(refine_logits, dim=0) if isinstance(refine_logits,(list,tuple)) else refine_logits
        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        
        obj_loss = F.cross_entropy(refine_obj_logits, fg_labels.long())
        
        # ********************************************************************
        
        return rel_loss,obj_loss
      
    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None,**kwargs):
        current_device,add_losses,add_data=torch.device(f'cuda:{torch.cuda.current_device()}'),dict(),dict()
        
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        # refine object labels        
        if self.mode == 'predcls' or self.mode=='sgdet':
            entity_dists, entity_preds, pos_embeds = self.refine_obj_labels(roi_features, proposals)
            entity_vis_rep=self.p_entity(roi_features)
            pos_embeds=pos_embeds.split(num_objs,dim=0)
        elif self.mode=='sgcls':
            entity_dists, entity_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)
            entity_vis_rep=F.relu(self.post_emb(edge_ctx))
            pos_embeds=[None]*len(num_objs)
            
            if self.training:
                binary_loss = []
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] =add_losses.get("binary_loss",0.0) + (sum(binary_loss) / len(binary_loss))
        else:
            raise ValueError(f'Unknow mode: {self.mode}') 
        
        entity_vis_rep = entity_vis_rep.view(entity_vis_rep.size(0), 2, self.hidden_dim) # entity representation
        
        sub_vis_reps = entity_vis_rep[:, 1].contiguous().view(-1, self.hidden_dim).split(num_objs,dim=0)
        obj_vis_reps = entity_vis_rep[:, 0].contiguous().view(-1, self.hidden_dim).split(num_objs,dim=0)
        
        entity_dists = entity_dists.split(num_objs, dim=0)
        entity_sem_reps= self.obj_embed(entity_preds.long()).split(num_objs,dim=0)
        entity_preds= entity_preds.split(num_objs, dim=0)
        # union_features=union_features.split(num_rels,dim=0)
        
        rel_sem_vector=self.p_pred(self.rel_embed.weight)
        rel_vis_reps,sub_sem_reps,obj_sem_reps,img_reps,pair_preds=[],[],[],[],[]
        for batch_idx,(proposal,sub_vis_rep,obj_vis_rep,entity_sem_rep,rel_pair_idx,entity_pred,pos_embed) in enumerate(zip(proposals,sub_vis_reps,obj_vis_reps,entity_sem_reps,rel_pair_idxs,entity_preds,pos_embeds)):
            image = Image.open(proposal.get_field('file_name'))
            image_inputs = self.clip_processor(images=image, return_tensors="pt").to(current_device)
            img_encode_out=self.clip_vision_model(**image_inputs)
            img_rep = self.align_img(img_encode_out.last_hidden_state[:,1:,:])  # without cls token
            
            if self.mode == 'predcls' or self.mode=='sgdet':
                sub_pos_embed,obj_pos_embed=self.p_pos(pos_embed[rel_pair_idx[:,0]]),self.p_pos(pos_embed[rel_pair_idx[:,1]])
            sub_vis_rep,obj_vis_rep=sub_vis_rep[rel_pair_idx[:,0]],obj_vis_rep[rel_pair_idx[:,1]]
            sub_sem_rep,obj_sem_rep=entity_sem_rep[rel_pair_idx[:,0]],entity_sem_rep[rel_pair_idx[:,1]]
        
            # ********************************************* refine visual features ***************************************************
            if self.mode == 'predcls' or self.mode=='sgdet':
                sub_geo_rep,obj_geo_rep=sub_vis_rep+F.relu(sub_pos_embed),obj_vis_rep+F.relu(obj_pos_embed)
            else:
                sub_geo_rep,obj_geo_rep=sub_vis_rep,obj_vis_rep
            rel_vis_rep,geo_vis_rep=self.rel_quary.expand(sub_geo_rep.shape[0],1,-1),torch.stack([sub_geo_rep,obj_geo_rep],dim=1)

            for (s_attn,s_norm,c_attn,c_norm,ffn,ffn_norm) in self.rel_query_init:
                attn_output, _ =s_attn(query=rel_vis_rep,key=rel_vis_rep,value=rel_vis_rep)
                rel_vis_rep=s_norm(rel_vis_rep+attn_output)
                
                attn_output, _ =c_attn(query=rel_vis_rep,key=geo_vis_rep,value=geo_vis_rep)
                rel_vis_rep=c_norm(rel_vis_rep+attn_output)
                
                rel_vis_rep=ffn_norm(ffn(rel_vis_rep)+rel_vis_rep)
            
            expand_img_rep=img_rep.expand(sub_geo_rep.shape[0],-1,-1)
            for (s_attn,s_norm,c_attn,c_norm,ffn,ffn_norm) in self.rel_query_refine:
                attn_output, _ =s_attn(query=rel_vis_rep,key=rel_vis_rep,value=rel_vis_rep)
                rel_vis_rep=s_norm(rel_vis_rep+attn_output)
                
                attn_output, _ =c_attn(query=rel_vis_rep,key=expand_img_rep,value=expand_img_rep)
                rel_vis_rep=c_norm(rel_vis_rep+attn_output)
                
                rel_vis_rep=ffn_norm(ffn(rel_vis_rep)+rel_vis_rep)

            rel_vis_rep=rel_vis_rep.squeeze(1)  # rel_num, hidden_dim
            rel_vis_reps.append(rel_vis_rep)
            # ********************************************* refine semantic features ***************************************************
            # refine object semantic features
            sub_sem_rep,obj_sem_rep=self.p_sub(sub_sem_rep),self.p_obj(obj_sem_rep)
            vis2sem_sub,vis2sem_obj,vis2sem_img=self.vis2sem(sub_vis_rep),self.vis2sem(obj_vis_rep),self.vis2sem(img_rep)
            
            gate_sub=F.sigmoid(self.gate_sub(torch.cat([sub_sem_rep,vis2sem_sub],dim=-1)))
            gate_obj=F.sigmoid(self.gate_obj(torch.cat([obj_sem_rep,vis2sem_obj],dim=-1)))
            
            sub_sem_rep,obj_sem_rep=sub_sem_rep+vis2sem_sub*gate_sub,obj_sem_rep+vis2sem_obj*gate_obj
            sub_sem_reps.append(sub_sem_rep)
            obj_sem_reps.append(obj_sem_rep)
            
            img_reps.append(vis2sem_img.expand(sub_geo_rep.shape[0],-1,-1))
            pair_preds.append(torch.stack([entity_pred[rel_pair_idx[:,0]],entity_pred[rel_pair_idx[:,1]]],dim=1))
        geo_rel_pre=self.geo_rel_pre(torch.cat(rel_vis_reps,dim=0))
        
        # refine predicate semantic features
        sub_sem_reps,obj_sem_reps=torch.cat(sub_sem_reps,dim=0),torch.cat(obj_sem_reps,dim=0)
        fusion_entity=F.relu(sub_sem_reps+obj_sem_reps)-(sub_sem_reps-obj_sem_reps)**2
        union_sem_reps=self.vis2sem(self.sample_union_rep(union_features))
        gate_union=F.sigmoid(self.gate_pred(torch.cat([fusion_entity,union_sem_reps],dim=-1)))
        
        rel_sem_reps=fusion_entity-union_sem_reps*gate_union
        rel_sem_reps=self.filter_rel_norm(self.filter_rel_rep(rel_sem_reps)+rel_sem_reps)
        rel_sem_reps=self.drop_rel_rep(rel_sem_reps)
        
        # refine triple semantic using image
        triple_sem_reps=torch.stack([sub_sem_reps,rel_sem_reps,obj_sem_reps],dim=1)
        img_reps=torch.cat(img_reps,dim=0)
        
        union_sem_reps,sem_rel_query=union_sem_reps.unsqueeze(1),self.sem_rel_quary.expand(img_reps.shape[0],1,-1)
        for (s_attn,s_norm,ffn,ffn_norm) in self.refine_union_vis:
            attn_output, _ =s_attn(query=img_reps,key=union_sem_reps,value=union_sem_reps)
            img_reps=s_norm(img_reps+attn_output)
            
            img_reps=ffn_norm(ffn(img_reps)+img_reps)
        
        for (s_attn,s_norm,c_attn,c_norm,ffn,ffn_norm) in self.init_sem_rel_query:
            s_attn_output, _= s_attn(query=sem_rel_query,key=sem_rel_query,value=sem_rel_query)
            sem_rel_query=s_norm(sem_rel_query+s_attn_output)
            
            attn_output, _ =c_attn(query=sem_rel_query,key=triple_sem_reps,value=triple_sem_reps)
            sem_rel_query=c_norm(sem_rel_query+attn_output)
            
            sem_rel_query=ffn_norm(ffn(sem_rel_query)+sem_rel_query)
        
        for (s_attn,s_norm,c_attn,c_norm,ffn,ffn_norm) in self.refine_sem_rel_query:
            s_attn_output, _= s_attn(query=sem_rel_query,key=sem_rel_query,value=sem_rel_query)
            sem_rel_query=s_norm(sem_rel_query+s_attn_output)
            
            attn_output, _ =c_attn(query=sem_rel_query,key=img_reps,value=img_reps)
            sem_rel_query=c_norm(sem_rel_query+attn_output)
            
            sem_rel_query=ffn_norm(ffn(sem_rel_query)+sem_rel_query)
        
        sem_rel_pre=self.sem_rel_pre(sem_rel_query.squeeze(1))
        
        # triple semantic similarity
        triple_query_sem_reps=torch.cat([sub_sem_reps,sem_rel_query.squeeze(1),obj_sem_reps],dim=-1)  
        triple_query_sem_reps=self.fusion_triple_sem_rep(triple_query_sem_reps)
        triple_sem_rel_pre=self.triple_sem_pre(triple_query_sem_reps)
        
        # semantic similarity
        rel_sem_vec=self.proj_head(self.drop_rel_rep(rel_sem_vector))
        rel_sem_reps=self.proj_head(rel_sem_reps)
        
        rel_sem_reps_norm = rel_sem_reps / rel_sem_reps.norm(dim=1, keepdim=True)  # r_norm
        rel_sem_vec_norm = rel_sem_vec / rel_sem_vec.norm(dim=1, keepdim=True)  # c_norm

        sem_rel_sim=rel_sem_reps_norm @ rel_sem_vec_norm.t() * self.logit_scale.exp()
        # final predicate dists
        if self.mode=='sgcls':
            rel_dists=geo_rel_pre*self.geo_rel_w+sem_rel_pre*self.sem_rel_w+sem_rel_sim*self.sem_rel_sim_w+triple_sem_rel_pre*self.triple_sem_rel_w
        else:
            rel_dists=geo_rel_pre+sem_rel_pre+sem_rel_sim+triple_sem_rel_pre
        
        if self.use_bias:
            freq_dist=self.freq_bias.index_with_labels(torch.cat(pair_preds,dim=0).long())
            rel_dists=rel_dists+freq_dist*getattr(self,'freq_weights',1)
        
        if self.training:
            rel_labels=torch.cat(rel_labels,dim=0)

            add_losses=self.calculate_similar_loss(rel_sem_vec,rel_sem_reps,rel_labels,add_losses,loss_name="loss_dis")
            
            add_data['final_loss']=dict()
            loss_relation,loss_refine=self.calculate_loss(proposals=proposals,refine_logits=entity_dists,relation_logits=rel_dists,rel_labels=rel_labels)
            add_data['final_loss']['loss_relation'],add_data['final_loss']['loss_refine']=loss_relation,loss_refine
        
        rel_dists=rel_dists.split(num_rels,dim=0)
        return entity_dists, rel_dists, add_losses, add_data
    
    def calculate_semantic_loss(self,semantic_feature,semantic_feature_norm):
        add_losses=dict()
        
        ### Prototype Regularization  ---- cosine similarity
        target_rpredicate_proto_norm = semantic_feature_norm.clone().detach() 
        simil_mat = semantic_feature_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
        l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (51*51)  
        add_losses.update({"l21_loss": l21})  # Le_sim = ||S||_{2,1}
        ### end
        
        ### Prototype Regularization  ---- Euclidean distance
        gamma2 = 7.0
        predicate_proto_a = semantic_feature.unsqueeze(dim=1).expand(-1, 51, -1) 
        predicate_proto_b = semantic_feature.detach().unsqueeze(dim=0).expand(51, -1, -1)
        proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
        sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
        topK_proto_dis = sorted_proto_dis_mat[:, :11].sum(dim=1) / 10   # obtain d-, where k2 = 1
        dist_loss = torch.max(torch.zeros(51).cuda(), -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
        add_losses.update({"dist_loss2": dist_loss})
        ### end
        
        return add_losses
        
    def calculate_similar_loss(self,semantic_feature,rel_rep,rel_labels,add_losses,loss_name="loss_dis"):
        ###  Prototype-based Learning  ---- Euclidean distance
        # rel_labels = cat(rel_labels, dim=0) if isinstance(rel_labels,(list,tuple)) else rel_labels
        gamma1 = 1.0
        rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, semantic_feature.shape[0], -1)  # r
        predicate_proto_expand = semantic_feature.unsqueeze(dim=0).expand(rel_rep.size(0), -1, -1)  # ci
        distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2    # Distance Set G, gi = ||r-ci||_2^2
        mask_neg = torch.ones(rel_rep.size(0), semantic_feature.shape[0]).cuda()  
        mask_neg[torch.arange(rel_rep.size(0)), rel_labels] = 0
        distance_set_neg = distance_set * mask_neg
        distance_set_pos = distance_set[torch.arange(rel_rep.size(0)), rel_labels]  # gt i.e., g+
        sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
        topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10  # obtaining g-, where k1 = 10, 
        loss_sum = torch.max(torch.zeros(rel_rep.size(0)).cuda(), distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
        add_losses[loss_name]=add_losses.get(loss_name,0.0)+loss_sum     # Le_euc = max(0, (g+) - (g-) + gamma1)
        ### end 
        
        return add_losses
    
    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        pos_embed = self.pos_embed(encode_box_info(proposals))

        # label/logits embedding will be used as input
        if self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed.weight

        assert proposals[0].mode == 'xyxy'

        pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred)  # 512 -> 151
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs).long()
            else:
                obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()
        
        return obj_dists, obj_preds, pos_embed

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


class PE_DPPLML(nn.Module):
    def __init__(self, config, in_channels):
        super(PE_DPPLML, self).__init__()

        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.cfg = config

        assert in_channels is not None
        self.in_channels = in_channels
        self.obj_dim = in_channels
        

        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        statistics = get_dataset_statistics(config)

        obj_classes, rel_classes,fg_matrix = statistics['obj_classes'], statistics['rel_classes'],statistics['fg_matrix']
        
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)
        
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM 
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.mlp_dim = 2048 # config.MODEL.ROI_RELATION_HEAD.PENET_MLP_DIM
        self.post_emb = nn.Linear(self.obj_dim, self.mlp_dim * 2)  

        self.embed_dim = 300 # config.MODEL.ROI_RELATION_HEAD.PENET_EMBED_DIM
        dropout_p = 0.2 # config.MODEL.ROI_RELATION_HEAD.PENET_DROPOUT
        
        
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)  # load Glove for objects
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)   # load Glove for predicates
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
       
        self.W_sub = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)

        self.gate_sub = nn.Linear(self.mlp_dim*2, self.mlp_dim)  
        self.gate_obj = nn.Linear(self.mlp_dim*2, self.mlp_dim)
        self.gate_pred = nn.Linear(self.mlp_dim*2, self.mlp_dim)

        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.mlp_dim, self.mlp_dim*2), nn.ReLU(True),
            nn.Dropout(dropout_p), nn.Linear(self.mlp_dim*2, self.mlp_dim)
        ])

        self.project_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim*2, 2)

        self.linear_sub = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_obj = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_pred = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_rel_rep = nn.Linear(self.mlp_dim, self.mlp_dim)
        
        self.norm_sub = nn.LayerNorm(self.mlp_dim)
        self.norm_obj = nn.LayerNorm(self.mlp_dim)
        self.norm_rel_rep = nn.LayerNorm(self.mlp_dim)

        self.dropout_sub = nn.Dropout(dropout_p)
        self.dropout_obj = nn.Dropout(dropout_p)
        self.dropout_rel_rep = nn.Dropout(dropout_p)
        
        self.dropout_rel = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)
       
        self.down_samp = MLP(self.pooling_dim, self.mlp_dim, self.mlp_dim, 2) 

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        ##### refine object labels
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.obj_dim = in_channels
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_classes) 
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'
        
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        # ****************************** Dynamic feature center ******************************
        num_head = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        dropout_rate = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        rel_layer = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER
        inner_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        
        self.proj_sub=make_fc(self.mlp_dim,self.hidden_dim)
        self.proj_obj=make_fc(self.mlp_dim,self.hidden_dim)
        self.proj_pred=make_fc(self.mlp_dim*2,self.hidden_dim)
        
        # self.rel_center=nn.Parameter(torch.tensor(self.rel_embed.weight))
        self.rel_query=nn.Parameter(torch.normal(mean=0, std=0.1, size=(self.hidden_dim,)))
        self.rel_query_init=nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.Sequential(
                    nn.Linear(self.hidden_dim,inner_dim),
                    nn.ReLU(),
                    nn.Linear(inner_dim,self.hidden_dim),
                    nn.Dropout(dropout_rate)
                ),
                nn.LayerNorm(self.hidden_dim)
            ]) for _ in range(rel_layer)
        ])
        
        self.rel_center_refine=nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.Sequential(
                    nn.Linear(self.hidden_dim,inner_dim),
                    nn.ReLU(),
                    nn.Linear(inner_dim,self.hidden_dim),
                    nn.Dropout(dropout_rate)
                ),
                nn.LayerNorm(self.hidden_dim),
                nn.Sequential(
                    nn.Linear(self.hidden_dim,inner_dim),
                    nn.ReLU(),
                    nn.Linear(inner_dim,self.hidden_dim),
                    nn.Dropout(dropout_rate)
                ),
                nn.LayerNorm(self.hidden_dim)
            ]) for _ in range(rel_layer)
        ])
        
        self.rel_weight=nn.Parameter(torch.ones((self.num_rel_cls,)))
        
        self.use_pcr=config.MODEL.ROI_RELATION_HEAD.USE_PCR
        if self.use_pcr:
            # ***************** entity: subject/object - predicate similarity *****************
            self.s_p,self.o_p=nn.Parameter(torch.normal(mean=0, std=0.1, size=(self.hidden_dim,))),nn.Parameter(torch.normal(mean=0, std=0.1, size=(self.hidden_dim,)))
            
            self.direct_pred_encoder=nn.ModuleList([
                nn.ModuleList([
                    nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                    nn.LayerNorm(self.hidden_dim),
                    nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                    nn.LayerNorm(self.hidden_dim),
                    nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                    nn.LayerNorm(self.hidden_dim),
                    nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                    nn.LayerNorm(self.hidden_dim),
                    nn.Sequential(
                        nn.Linear(self.hidden_dim,inner_dim),
                        nn.ReLU(),
                        nn.Linear(inner_dim,self.hidden_dim),
                        nn.Dropout(dropout_rate)
                    ),
                    nn.LayerNorm(self.hidden_dim),
                    nn.Sequential(
                        nn.Linear(self.hidden_dim,inner_dim),
                        nn.ReLU(),
                        nn.Linear(inner_dim,self.hidden_dim),
                        nn.Dropout(dropout_rate)
                    ),
                    nn.LayerNorm(self.hidden_dim)
                ]) for _ in range(rel_layer)
            ])
            
            self.s_p_o_weight=nn.Parameter(torch.ones((self.num_rel_cls,)))
            
            # ***************** entity: subject/object - predicate similarity *****************
            self.refine_double_predicate=nn.ModuleList([
                nn.ModuleList([
                    nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                    nn.LayerNorm(self.hidden_dim),
                    nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                    nn.LayerNorm(self.hidden_dim),
                    nn.Sequential(
                        nn.Linear(self.hidden_dim,inner_dim),
                        nn.ReLU(),
                        nn.Linear(inner_dim,self.hidden_dim),
                        nn.Dropout(dropout_rate)
                    ),
                    nn.LayerNorm(self.hidden_dim),
                    nn.Sequential(
                        nn.Linear(self.hidden_dim,inner_dim),
                        nn.ReLU(),
                        nn.Linear(inner_dim,self.hidden_dim),
                        nn.Dropout(dropout_rate)
                    ),
                    nn.LayerNorm(self.hidden_dim),
                ]) for _ in range(rel_layer)
            ])
            self.refine_sem_query=nn.ModuleList([
                nn.ModuleList([
                    nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                    nn.LayerNorm(self.hidden_dim),
                    nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                    nn.LayerNorm(self.hidden_dim),
                    nn.Sequential(
                        nn.Linear(self.hidden_dim,inner_dim),
                        nn.ReLU(),
                        nn.Linear(inner_dim,self.hidden_dim),
                        nn.Dropout(dropout_rate)
                    ),
                    nn.LayerNorm(self.hidden_dim),
                ]) for _ in range(rel_layer)
            ])
        
        # **************** loss ********************
        self.gamma,self.total_iters=1,config.SOLVER.MAX_ITER
        bata=0.9999
        
        per_predicate_num=np.sum(fg_matrix.numpy(),axis=(0,1))
        self.per_predicate_weight=torch.tensor([(1-bata)/(1-bata**pre_num) for pre_num in per_predicate_num],dtype=torch.float)
        self.rel_ce_loss=nn.CrossEntropyLoss(self.per_predicate_weight)
        
        # *************** predicator bias *******************
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)
            self.freq_weight=nn.Parameter(torch.ones((self.num_rel_cls,)))
        self.ori_rel_weight=nn.Parameter(torch.ones((self.num_rel_cls,)))
        
    def calculate_loss(self,relation_logits,rel_labels,proposals=None,refine_logits=None):
        # ************************ relation loss ****************************
        relation_logits,rel_labels=torch.cat(relation_logits,dim=0) if isinstance(relation_logits,(list,tuple)) else relation_logits,torch.cat(rel_labels,dim=0) if isinstance(rel_labels,(list,tuple)) else rel_labels
        rel_ce_loss=self.rel_ce_loss(relation_logits,rel_labels)
        
        rel_log_softmax = torch.log_softmax(relation_logits, dim=1)
        rel_logpt = torch.gather(rel_log_softmax, dim=1, index=rel_labels.view(-1, 1)).view(-1)
        
        rel_loss=(1-torch.exp(rel_logpt))**self.gamma*rel_ce_loss
        rel_loss=torch.mean(rel_loss)  # torch.sum(f_loss)
        
        # **************************** object loss ***************************
        if proposals is not None and refine_logits is not None:
            refine_obj_logits = cat(refine_logits, dim=0) if isinstance(refine_logits,(list,tuple)) else refine_logits
            fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            
            obj_loss = F.cross_entropy(refine_obj_logits, fg_labels.long())
        else:
            obj_loss= None
        # ********************************************************************
        
        return rel_loss,obj_loss
    
    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):

        add_losses = {}
        add_data = {}

        # refine object labels
        entity_dists, entity_preds = self.refine_obj_labels(roi_features, proposals)
        ##### 

        entity_rep = self.post_emb(roi_features)   # using the roi features obtained from the faster rcnn
        entity_rep = entity_rep.view(entity_rep.size(0), 2, self.mlp_dim)

        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)    # xs
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)    # xo

        entity_embeds = self.obj_embed(entity_preds) # obtaining the word embedding of entities with GloVe 

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        sub_reps = sub_rep.split(num_objs, dim=0)
        obj_reps = obj_rep.split(num_objs, dim=0)
        entity_preds = entity_preds.split(num_objs, dim=0)
        entity_embeds = entity_embeds.split(num_objs, dim=0)

        fusion_so = []
        pair_preds = []
        sub_embeds,obj_embeds=[],[]

        for pair_idx, sub_rep, obj_rep, entity_pred, entity_embed, proposal in zip(rel_pair_idxs, sub_reps, obj_reps, entity_preds, entity_embeds, proposals):

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
            pair_preds.append(torch.stack((entity_pred[pair_idx[:, 0]], entity_pred[pair_idx[:, 1]]), dim=1))

            sub_embeds.append(sub)
            obj_embeds.append(obj)
        fusion_so = cat(fusion_so, dim=0)  
        pair_pred = cat(pair_preds, dim=0) 

        sem_pred = self.vis2sem(self.down_samp(union_features))  # h(xu)
        gate_sem_pred = torch.sigmoid(self.gate_pred(cat((fusion_so, sem_pred), dim=-1)))  # gp

        rel_rep = fusion_so - sem_pred * gate_sem_pred  #  F(s,o) - gp · h(xu)   i.e., r = F(s,o) - up
        predicate_proto = self.W_pred(self.rel_embed.weight)  # c = Wp x tp  i.e., semantic prototypes
        
        ##### for the model convergence
        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)

        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))
        ######
        
        rel_cen_dists,extra_dists,add_losses,rel_center_features=self.update_rel_center(torch.cat(sub_embeds,dim=0),torch.cat(obj_embeds,dim=0),rel_rep,predicate_proto,rel_labels,add_losses,proposals,rel_pair_idxs,rel_nums=num_rels)

        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  #  <r_norm, c_norm> / τ
        # the rel_dists will be used to calculate the Le_sim with the ce_loss
        
        rel_dists=rel_dists*self.ori_rel_weight+rel_cen_dists*self.rel_weight
        
        for key,value in extra_dists.items():
            rel_dists=rel_dists+value
        
        if self.use_bias:
            rel_dists=rel_dists+self.freq_bias.index_with_labels(pair_pred.long())*self.freq_weight

        if self.training:
            ### Prototype Regularization  ---- cosine similarity
            target_rpredicate_proto_norm = predicate_proto_norm.clone().detach() 
            simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
            l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (self.num_rel_cls*self.num_rel_cls)  
            add_losses.update({"l21_loss": l21})  # Le_sim = ||S||_{2,1}
            ### end
            
            ### Prototype Regularization  ---- Euclidean distance
            gamma2 = 7.0
            predicate_proto_a = predicate_proto.unsqueeze(dim=1).expand(-1, self.num_rel_cls, -1) 
            predicate_proto_b = predicate_proto.detach().unsqueeze(dim=0).expand(self.num_rel_cls, -1, -1)
            proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
            sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
            topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1   # obtain d-, where k2 = 1
            dist_loss = torch.max(torch.zeros(self.num_rel_cls).cuda(), -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
            add_losses.update({"dist_loss2": dist_loss})
            ### end 

            ###  Prototype-based Learning  ---- Euclidean distance
            rel_labels = cat(rel_labels, dim=0)
            gamma1 = 1.0
            rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, self.num_rel_cls, -1)  # r
            predicate_proto_expand = predicate_proto.unsqueeze(dim=0).expand(rel_labels.size(0), -1, -1)  # ci
            distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2    # Distance Set G, gi = ||r-ci||_2^2
            mask_neg = torch.ones(rel_labels.size(0), self.num_rel_cls).cuda()  
            mask_neg[torch.arange(rel_labels.size(0)), rel_labels] = 0
            distance_set_neg = distance_set * mask_neg
            distance_set_pos = distance_set[torch.arange(rel_labels.size(0)), rel_labels]  # gt i.e., g+
            sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
            topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10  # obtaining g-, where k1 = 10, 
            loss_sum = torch.max(torch.zeros(rel_labels.size(0)).cuda(), distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
            add_losses.update({"loss_dis": loss_sum})     # Le_euc = max(0, (g+) - (g-) + gamma1)
            ### end 

            add_data['final_loss']=dict()
            loss_relation,loss_refine=self.calculate_loss(relation_logits=rel_dists,rel_labels=rel_labels,proposals=proposals,refine_logits=entity_dists)
            add_data['final_loss']['loss_relation'],add_data['final_loss']['loss_refine']=loss_relation,loss_refine
        
        entity_dists = entity_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        return entity_dists, rel_dists, add_losses, add_data

    def update_rel_center(self,sub_embeds,obj_embeds,rel_reps,predicate_reps,rel_labels=None,add_losses=None,proposals=None,rel_pairs=None,rel_nums=-1):
        # dynamic update relation center
        # rel_reps shape: (batch_size, hidden_dim)
        
        # assert sub_embeds.shape[-1]==obj_embeds.shape[-1]==rel_reps.shape[-1], f'subject embeds shape: {sub_embeds.shape}, object embeds shape: {sub_embeds.shape}, relation representation: {rel_reps.shape}'
        sub_embeds,obj_embeds,rel_reps,predicate_reps=self.proj_sub(sub_embeds),self.proj_obj(obj_embeds),self.proj_pred(rel_reps),self.proj_pred(predicate_reps)
        
        tri_embeds=torch.stack([sub_embeds,rel_reps,obj_embeds],dim=1)
        sem_rel_querys=self.rel_query.expand(tri_embeds.shape[0],1,-1)
        
        # refine useful relation features
        for (s_attn,s_norm,c_attn,c_norm,ffn,ffn_norm) in self.rel_query_init:
            attn_output, _ =s_attn(query=sem_rel_querys,key=sem_rel_querys,value=sem_rel_querys)
            sem_rel_querys=s_norm(sem_rel_querys+attn_output)
            
            attn_output, _ =c_attn(query=sem_rel_querys,key=tri_embeds,value=tri_embeds)
            sem_rel_querys=c_norm(sem_rel_querys+attn_output)
            
            sem_rel_querys=ffn_norm(ffn(sem_rel_querys)+sem_rel_querys)
        sem_rel_querys=sem_rel_querys.squeeze(1) # sample_nums,hidden_dim
        
        # refine relation cluster center features
        rel_center_features,sem_rel_querys=predicate_reps.unsqueeze(0),sem_rel_querys.unsqueeze(0)
        for (s_attn,s_norm,c_attn,c_norm,ffn_cen,ffn_cen_norm,ffn_rep,ffn_rep_norm) in self.rel_center_refine:
            attn_output, _ =s_attn(query=rel_center_features,key=rel_center_features,value=rel_center_features)
            rel_center_features=s_norm(rel_center_features+attn_output)
            
            attn_output, cen_sim_rep =c_attn(query=rel_center_features,key=sem_rel_querys,value=sem_rel_querys)
            rel_center_features=c_norm(rel_center_features+attn_output)
            
            attn_output, rep_sim_cen =c_attn(query=sem_rel_querys,key=rel_center_features,value=rel_center_features)
            sem_rel_querys=c_norm(sem_rel_querys+attn_output)
            
            rel_center_features=ffn_cen_norm(ffn_cen(rel_center_features)+rel_center_features)
            sem_rel_querys=ffn_rep_norm(ffn_rep(sem_rel_querys)+sem_rel_querys)
        
        rel_center_features=rel_center_features.squeeze(0) #  num_rels,hidden_dim
        sem_rel_querys=sem_rel_querys.squeeze(0) #  sample_nums,hidden_dim
        
        if self.use_pcr:
            # ***************** entity: subject/object - predicate similarity *****************
            s_p_query,o_p_query=self.s_p.expand(tri_embeds.shape[0],1,-1),self.o_p.expand(tri_embeds.shape[0],1,-1)
            s_p_rep,o_p_rep=torch.stack([sub_embeds,rel_reps],dim=1),torch.stack([obj_embeds,rel_reps],dim=1)
            for (s_s_p_attn,s_s_p_norm,s_o_p_attn,s_o_p_norm,s_p_attn,s_p_norm,o_p_attn,o_p_norm,ffn_s_p,ffn_s_p_norm,ffn_o_p,ffn_o_p_norm) in self.direct_pred_encoder:
                # ************************** subject-predicate **************************
                
                attn_output, _ =s_s_p_attn(query=s_p_query,key=s_p_query,value=s_p_query)
                s_p_query=s_s_p_norm(s_p_query+attn_output)
                
                attn_output, _ =s_p_attn(query=s_p_query,key=s_p_rep,value=s_p_rep)
                s_p_query=s_p_norm(s_p_query+attn_output)
                
                s_p_query=ffn_s_p_norm(ffn_s_p(s_p_query)+s_p_query)
                
                # ************************** object-predicate **************************
                
                attn_output, _ =s_o_p_attn(query=o_p_query,key=o_p_query,value=o_p_query)
                o_p_query=s_o_p_norm(o_p_query+attn_output)
                
                attn_output, _ =o_p_attn(query=o_p_query,key=o_p_rep,value=o_p_rep)
                o_p_query=o_p_norm(o_p_query+attn_output)
                
                o_p_query=ffn_o_p_norm(ffn_o_p(o_p_query)+o_p_query)

            tri_rel_center_reps=rel_center_features.clone().detach().expand(s_p_query.shape[0],-1,-1)
            for (c_sp_attn,c_sp_norm,c_op_attn,c_op_norm,sp_ffn,sp_ffn_norm,op_ffn,op_ffn_norm) in self.refine_double_predicate:
                attn_output, sp_attn_weight =c_sp_attn(query=s_p_query,key=tri_rel_center_reps,value=tri_rel_center_reps)
                s_p_query=c_sp_norm(s_p_query+attn_output)   
                
                s_p_query=sp_ffn_norm(sp_ffn(s_p_query)+s_p_query)
                
                attn_output, op_attn_weight =c_op_attn(query=o_p_query,key=tri_rel_center_reps,value=tri_rel_center_reps)
                o_p_query=c_op_norm(o_p_query+attn_output)   
                
                o_p_query=op_ffn_norm(op_ffn(o_p_query)+o_p_query)
                
            # refine semantic relationship querys
            tri_predicate_reps,sem_rel_querys=torch.cat([s_p_query,o_p_query],dim=1),sem_rel_querys.unsqueeze(1)
            for (s_attn,s_norm,c_attn,c_norm,ffn,ffn_norm) in self.refine_sem_query:
                attn_output, _ =s_attn(query=tri_predicate_reps,key=tri_predicate_reps,value=tri_predicate_reps)
                tri_predicate_reps=s_norm(tri_predicate_reps+attn_output)   
                
                attn_output, _ =c_attn(query=sem_rel_querys,key=tri_predicate_reps,value=tri_predicate_reps)
                sem_rel_querys=c_norm(sem_rel_querys+attn_output)   
                
                sem_rel_querys=ffn_norm(ffn(sem_rel_querys)+sem_rel_querys)
            
            s_p_query,o_p_query,sem_rel_querys=tri_predicate_reps[:,0,:],tri_predicate_reps[:,1,:],sem_rel_querys.squeeze(1)
            # s_p_query,o_p_query,sem_rel_querys=s_p_query.squeeze(1),o_p_query.squeeze(1),sem_rel_querys.squeeze(1)
        
        if self.training:
            rel_labels=torch.cat(rel_labels,dim=0)
            # L_cd
            add_losses=self.extra_loss(sem_rel_querys,rel_center_features,rel_labels,predicate_reps,add_losses,loss_fun='intra_cls_loss',loss_name='intra_cls_loss')
            
            bi_rels=torch.zeros(sem_rel_querys.shape[0],self.num_rel_cls,device=torch.device(f'cuda:{torch.cuda.current_device()}'))
            bi_rels[torch.arange(rel_reps.shape[0]),rel_labels]=1
            # L_cs
            # add_losses['rep_attn_cen_loss']=add_losses.get('rep_attn_cen_loss',0.0)+F.mse_loss(rep_sim_cen.squeeze(0),bi_rels)
            
            if self.use_pcr:
                # L_spd
                add_losses=self.extra_loss(s_p_query,rel_center_features.detach(),rel_labels,predicate_reps,add_losses,loss_fun='intra_cls_loss',loss_name='sub_pred_rep_loss')
                # L_opd
                add_losses=self.extra_loss(o_p_query,rel_center_features.detach(),rel_labels,predicate_reps,add_losses,loss_fun='intra_cls_loss',loss_name='obj_pred_rep_loss')
                
                # L_pc
                # add_losses['sp_attn_loss']=add_losses.get('sp_attn_loss',0.0)+F.mse_loss(sp_attn_weight.squeeze(1),bi_rels)
                # add_losses['op_attn_loss']=add_losses.get('op_attn_loss',0.0)+F.mse_loss(op_attn_weight.squeeze(1),bi_rels)
            
            # add_losses['sub_obj_pred_dis']=add_losses.get('sub_obj_pred_dis',0.0)+F.mse_loss(s_p_query,o_p_query)
            # add_losses=self.extra_loss(s_p_query,o_p_query,_,predicate_reps,add_losses,loss_fun='inter_cls_loss',loss_name='sub_obj_pred_dis')
        
        # ********************** semantic relation query -- relation center distance **********************
        sem_rel_reps,rel_center_reps=sem_rel_querys.unsqueeze(dim=1).expand(-1,self.num_rel_cls,-1),rel_center_features.unsqueeze(dim=0).expand(sem_rel_querys.shape[0],-1,-1)
        dis_mat=(sem_rel_reps-rel_center_reps).norm(dim=2)**2
        
        dis_mat=1-dis_mat.softmax(dim=-1)
        
        """
        # ********************** subject-predicate query -- relation center distance **********************
        s_p_reps,rel_center_reps=s_p_query.unsqueeze(dim=1).expand(-1,self.num_rel_cls,-1),rel_center_features.unsqueeze(dim=0).expand(sem_rel_querys.shape[0],-1,-1)
        s_p_dis_mat=(s_p_reps-rel_center_reps).norm(dim=2)**2
        
        s_p_dis_mat=1-s_p_dis_mat.softmax(dim=-1)
        
        # ********************** subject-predicate query -- relation center distance **********************
        o_p_reps,rel_center_reps=o_p_query.unsqueeze(dim=1).expand(-1,self.num_rel_cls,-1),rel_center_features.unsqueeze(dim=0).expand(sem_rel_querys.shape[0],-1,-1)
        o_p_dis_mat=(o_p_reps-rel_center_reps).norm(dim=2)**2
        
        o_p_dis_mat=1-o_p_dis_mat.softmax(dim=-1)
        """
        
        return dis_mat,dict(),add_losses,rel_center_features
            
    def extra_loss(self,rel_reps,rel_center,rel_labels,predicate_reps,add_losses,loss_fun,loss_name,top_k=15):
        if 'cluster_loss' in loss_fun:
            gamma=7.0
            predicate_cen_a=rel_center.unsqueeze(1).expand(-1,self.num_rel_cls,-1)  # rel_cls,rel_cls,hidden_dim
            predicate_cen_b=rel_center.detach().unsqueeze(dim=0).expand(self.num_rel_cls,-1,-1)  # rel_cls,rel_cls,hidden_dim           

            distance=(predicate_cen_a-predicate_cen_b).norm(dim=2)**2
            sort_dis,_=torch.sort(distance,dim=1)
            
            min_dis_norm=sort_dis[:,1:top_k].sum(dim=1)/(top_k-1)
            dis_loss=torch.max(torch.zeros(self.num_rel_cls,device=torch.device(f'cuda:{torch.cuda.current_device()}')),-min_dis_norm+gamma).mean()
            add_losses[loss_name]=add_losses.get(loss_name,0.0)+dis_loss

        if 'intra_cls_loss' in loss_fun:
            gamma=1.0
            expand_rel_rep=rel_reps.unsqueeze(dim=1).expand(-1,self.num_rel_cls,-1) # sample_nums,rel_cls,hidden_dim
            expand_rel_center=rel_center.unsqueeze(dim=0).expand(rel_reps.shape[0],-1,-1) # sample_nums,rel_cls,hidden_dim
            
            rel_reps_dis_center=(expand_rel_rep-expand_rel_center).norm(dim=2)**2 
            neg_masks=torch.ones(rel_reps.shape[0],self.num_rel_cls,device=torch.device(f'cuda:{torch.cuda.current_device()}'))
            neg_masks[torch.arange(rel_reps.shape[0]),rel_labels]=0
            
            neg_dis=neg_masks*rel_reps_dis_center
            # sort_neg_dis,_=torch.sort(neg_dis,dim=1)
            neg_dis=neg_dis.sum(dim=1)/neg_dis.shape[0]
            
            pos_dis=rel_reps_dis_center[torch.arange(rel_reps.shape[0]),rel_labels]
            dis_loss=torch.max(torch.zeros(rel_reps.shape[0],device=torch.device(f'cuda:{torch.cuda.current_device()}')),pos_dis-neg_dis+gamma).mean()
            add_losses[loss_name]=add_losses.get(loss_name,0.0)+dis_loss
        
        if 'inter_cls_loss' in loss_fun:
            gamma=1.0
            expand_rel_rep=rel_reps.unsqueeze(dim=1).expand(-1,rel_center.shape[0],-1) # sample_nums,rel_cls,hidden_dim
            expand_rel_center=rel_center.unsqueeze(dim=0).expand(rel_reps.shape[0],-1,-1) # sample_nums,rel_cls,hidden_dim
            
            rel_reps_dis_center=(expand_rel_rep-expand_rel_center).norm(dim=2)**2 
            neg_masks=torch.ones(rel_reps.shape[0],rel_center.shape[0],device=torch.device(f'cuda:{torch.cuda.current_device()}'))
            neg_masks[torch.arange(rel_reps.shape[0]),torch.arange(rel_center.shape[0])]=0
            
            neg_dis=neg_masks*rel_reps_dis_center
            # sort_neg_dis,_=torch.sort(neg_dis,dim=1)
            neg_dis=neg_dis.sum(dim=1)/neg_dis.shape[0]
            
            pos_dis=rel_reps_dis_center[torch.arange(rel_reps.shape[0]),torch.arange(rel_center.shape[0])]
            dis_loss=torch.max(torch.zeros(rel_reps.shape[0],device=torch.device(f'cuda:{torch.cuda.current_device()}')),pos_dis-neg_dis+gamma).mean()
            add_losses[loss_name]=add_losses.get(loss_name,0.0)+dis_loss

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
    
   
class Multi_step_Denoise(nn.Module):
    """_summary_

    Args:
        nn (_type_): 基于Motifs等提取的特征进行聚类，并将不断优化聚类中心，同时引入KNN计算特征间相似度，最终基于聚类后的特征进行特征预测
    """
    def __init__(self, config, in_channels, statistics,baseline_model="PENet"):
        super().__init__()
        self.config=config
        self.logger=logging.getLogger(__name__)
        
        num_head = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        dropout_rate = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        rel_layer = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER
        inner_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        
        self.baseline_model=baseline_model
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.mlp_dim = in_channels
        self.k_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM         
        self.v_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM    
        
        self.embed_dim = 300 # config.MODEL.ROI_RELATION_HEAD.PENET_EMBED_DIM
        
        obj_classes, rel_classes,fg_matrix = statistics['obj_classes'], statistics['rel_classes'],statistics['fg_matrix']
        assert self.num_rel_cls == len(rel_classes)
        self.rel_classes = rel_classes
        
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)   # load Glove for predicates
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)   # load Glove for predicates
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        self.obj_embed = nn.Embedding(len(obj_classes), self.embed_dim)
        with torch.no_grad():
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.filter_pred_prot=nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.project_prot_head = MLP(self.mlp_dim, self.mlp_dim,self.hidden_dim,2)
        
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # contrast learning
        
        # *************************** generate predicate reps based on union and entity pair reps ***************************
        self.cps_t_sub_reps,self.cps_t_obj_reps=MLP(self.hidden_dim,self.mlp_dim//2,self.hidden_dim,1),MLP(self.hidden_dim,self.mlp_dim//2,self.hidden_dim,1)

        self.cps_entity_pair_reps,self.gate_vis_entity=MLP(2*self.hidden_dim,self.mlp_dim//2,self.hidden_dim,1),MLP(2*self.hidden_dim,self.mlp_dim//2,self.hidden_dim,1)
        
        # *************************** entity node pair --> predicate reps ***************************
        self.cps_union_reps=MLP(self.pooling_dim,self.mlp_dim//2,self.hidden_dim,1)
        self.use_node_branch=config.MODEL.ROI_RELATION_HEAD.USE_NODE_BRANCH
        if self.use_node_branch:
            self.edge_rel_reps=nn.Parameter(torch.normal(mean=0, std=0.1, size=(self.hidden_dim,)))
            self.node_to_pre=nn.ModuleList([
                nn.ModuleList([
                    Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate), # Enhance Node
                    Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate), # Enhance Node
                    nn.LayerNorm(self.hidden_dim),
                    nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),  # generate predicate reps
                    nn.LayerNorm(self.hidden_dim),
                    MLP(self.hidden_dim,self.mlp_dim//2,self.hidden_dim,1),  # proj predicate reps -> predicate prototype 
                    Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate) # Cross attention predicate prototype           
                ]) for _  in range(rel_layer)
            ])
            
            self.refine_edge_pre=nn.ModuleList([
                nn.ModuleList([
                    nn.ModuleList([
                        nn.LayerNorm(self.hidden_dim),
                        nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True), 
                        nn.LayerNorm(self.hidden_dim),
                        MLP(self.hidden_dim,self.mlp_dim//2,self.hidden_dim,1), # Enhance entity weight in union features
                    ]),
                    nn.Sequential(
                        nn.Linear(2*self.hidden_dim,self.hidden_dim),
                        nn.Sigmoid()
                    ),  # del entity features
                    Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate), # Enhance predicate prototye reps in union reps
                    Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate)  # Refine predicate reps
                ]) for _ in range(rel_layer)
            ])

            self.logger.info('init node relation representation branch......')
            self.edg_rel_sim_emp_weight,self.pos_edg_rel_sim_scores,self.neg_edg_rel_sim_scores=torch.ones(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False)
            
        # *************************** union triple --> predicate reps ***************************
        self.use_denoise_branch=config.MODEL.ROI_RELATION_HEAD.USE_DENOISE_BRANCH
        if self.use_denoise_branch:
            self.noise_factor=nn.Parameter(torch.ones(1),requires_grad=True)  # add noise
            
            self.denoise_modules=nn.ModuleList([
                nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.hidden_dim,self.hidden_dim),
                        nn.LayerNorm(self.hidden_dim),
                        nn.Linear(self.hidden_dim,self.hidden_dim),
                        nn.ReLU(inplace=True)
                    ), # denoise
                    nn.ModuleList([  # denoise subject/object features
                        Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate), # subject self attention
                        Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate), # object self attention
                        nn.Sequential(
                            nn.Linear(2*self.hidden_dim,self.hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(self.hidden_dim,self.hidden_dim)
                        ),
                        Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate), # predicate reps attention sub-obj reps
                        nn.Sequential(
                            nn.Linear(2*self.hidden_dim,self.hidden_dim),
                            nn.Sigmoid()
                        ),  # del entity features
                        Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate) # t_predicate reps attention union features
                    ])
                ]) for _ in range(rel_layer)
            ])
            
            self.denoise_reps=nn.Sequential(
                        nn.Linear(self.hidden_dim,self.hidden_dim),
                        nn.LayerNorm(self.hidden_dim),
                        nn.Linear(self.hidden_dim,self.hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2)
                    )
            
            self.logger.info('init denoise relation representation branch......')
            self.recon_rel_sim_emp_weight,self.pos_recon_rel_sim_scores,self.neg_recon_rel_sim_scores=torch.ones(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False)
            
        # **************** Semantic consistency module ****************
        self.align_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim*2, 2)
        self.filter_noise_rel=nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # **************** discriminator module ****************
        self.step=config.MODEL.ROI_RELATION_HEAD.TRAIN_STEP
        if self.step!=1:
            self.build_diff_modules()
                
        self.use_branch_fusion=config.MODEL.ROI_RELATION_HEAD.USE_BRANCH_FUSION
        if self.use_branch_fusion and self.use_denoise_branch and self.use_node_branch:
            self.filter_tri=nn.Sequential(
                nn.Sigmoid(),
                nn.Dropout(0.2)
            )
            self.filter_recon=nn.Sequential(
            nn.Sigmoid(),
            nn.Dropout(0.2)
        )   

            self.logger.info('init fusion node and denoise reps branch......')
            self.sum_rel_sim_emp_weight,self.pos_sum_rel_sim_scores,self.neg_sum_rel_sim_scores=torch.ones(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False)
        
        # **************** process global features ****************
        self.use_global_vis_refine=config.MODEL.ROI_RELATION_HEAD.USE_GLOBAL_VISUAL
        if self.use_global_vis_refine:
            self.ds_glob_reps=nn.Sequential(
                nn.Linear(5*config.MODEL.RESNETS.BACKBONE_OUT_CHANNELS,self.hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim,self.hidden_dim)
            )
            
            self.proj_glob_reps = MLP(self.hidden_dim, self.hidden_dim, self.mlp_dim*2, 2)
            self.filter_glob_reps=nn.Sequential(
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
            self.glob_refine_rel_reps=nn.ModuleList([
                nn.ModuleList([
                    Trans_block(1,num_head,self.k_dim,self.v_dim,self.mlp_dim*2,self.hidden_dim,dropout_rate) if self.use_node_branch else nn.Identity(),  # for edg rel reps
                    Trans_block(1,num_head,self.k_dim,self.v_dim,self.mlp_dim*2,self.hidden_dim,dropout_rate) if self.use_denoise_branch else nn.Identity(),  # for denoise rel reps
                    Trans_block(1,num_head,self.k_dim,self.v_dim,self.mlp_dim*2,self.hidden_dim,dropout_rate) if self.use_branch_fusion and self.use_denoise_branch and self.use_node_branch else nn.Identity(),  # for sum rel reps
                    ]) for _ in range(rel_layer)
                ])
        
        self.use_global_rel_reps=config.MODEL.ROI_RELATION_HEAD.USE_GLOBAL_REPRESENTATION
        if self.use_global_rel_reps and self.use_denoise_branch and self.use_node_branch:
            self.global_rel_reps=nn.Parameter(torch.normal(mean=0, std=0.1, size=(self.mlp_dim*2,)))
            self.merge_rel_reps=nn.ModuleList([
                nn.ModuleList([
                    nn.LayerNorm(self.mlp_dim*2),
                    nn.MultiheadAttention(self.mlp_dim*2,num_head,dropout=dropout_rate,batch_first=True),
                    nn.LayerNorm(self.mlp_dim*2),
                    nn.MultiheadAttention(self.mlp_dim*2,num_head,dropout=dropout_rate,batch_first=True),
                    nn.LayerNorm(self.mlp_dim*2),
                    nn.MultiheadAttention(self.mlp_dim*2,num_head,dropout=dropout_rate,batch_first=True), 
                    nn.LayerNorm(self.mlp_dim*2),
                    MLP(self.mlp_dim*2,self.mlp_dim//2,self.mlp_dim*2,1)
                ]) for _ in range(rel_layer)
            ])
        
        self.use_kl_modules=config.MODEL.ROI_RELATION_HEAD.USE_KL_MODULE
        if self.use_kl_modules:
            self.kl_infos=nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.mlp_dim*2,self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim,self.mlp_dim),
                    nn.LayerNorm(self.mlp_dim),
                    nn.ReLU(),
                    nn.Linear(self.mlp_dim,self.mlp_dim*2)
                ),  # mean
                nn.Sequential(
                    nn.Linear(self.mlp_dim*2,self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim,self.mlp_dim),
                    nn.LayerNorm(self.mlp_dim),
                    nn.ReLU(),
                    nn.Linear(self.mlp_dim,self.mlp_dim*2)
                )  # log std
            ])
                
        # self.predict_method=config.MODEL.ROI_RELATION_HEAD.PRE_RESULT
        # assert self.predict_method=="sum" or ( self.predict_method==None and self.use_glob_refine_modules ), print(f'if predict method is not sum, please check using glob refine module')
        # ******************** loss ********************
        self.gamma,self.total_iters=1,config.SOLVER.MAX_ITER
        bata=0.9999
        
        per_predicate_num=np.sum(fg_matrix.numpy(),axis=(0,1))
        self.per_predicate_weight=torch.tensor([(1-bata)/(1-bata**pre_num) for pre_num in per_predicate_num],dtype=torch.float)
        self.rel_ce_loss=nn.CrossEntropyLoss(self.per_predicate_weight)
        
        self.use_adaptive_loss=config.MODEL.ROI_RELATION_HEAD.USE_ADAPTIVE_REWEIGHT_LOSS
        self.gt_scores,self.emp_decay=torch.zeros(self.num_rel_cls,requires_grad=False),0.8
    
    def build_diff_modules(self,flow_depth=14):
        self.enc_mean_std=nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.mlp_dim*2,self.mlp_dim),
                nn.LayerNorm(self.mlp_dim),
                nn.ReLU(),
                nn.Linear(self.mlp_dim,self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim,self.mlp_dim*2)
            ),  # proj mean
            nn.Sequential(
                nn.Linear(self.mlp_dim*2,self.mlp_dim),
                nn.LayerNorm(self.mlp_dim),
                nn.ReLU(),
                nn.Linear(self.mlp_dim,self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim,self.mlp_dim*2)
            )   # proj std
        ])
        
        self.previous_res()
        from .diffusion_utils import flow_model,diffusion_model
        # self.flow_module=flow_model(self.mlp_dim*2,self.mlp_dim*2,flow_depth)
        self.diff_module=diffusion_model(self.mlp_dim*2)
        self.tail_weight=nn.Parameter(torch.ones(self.num_rel_cls),requires_grad=True)
        
        self.correct_counts=torch.zeros(self.num_rel_cls)
        self.total_counts=torch.zeros(self.num_rel_cls)

    def freeze_module(self):
        pass
    
    def previous_res(self):
        logger=logging.getLogger(__name__)
        if self.step!=1:
            pre_step_res=torch.load(f'{os.path.dirname(self.config.MODEL.PRETRAINED_DETECTOR_CKPT)}/recall.pt',map_location='cpu')
            logger.info(f'load previous predicate recall score success, recall info: {pre_step_res}')
            self.previou_rel_score=[pre_step_res[rel_name] if rel_name in pre_step_res.keys() else 1.0  for rel_name in self.rel_classes]
            self.high_conf_cls=(torch.tensor(self.previou_rel_score)>0.5).int()
        else:
            logger.warning('load previous recall score failed........')
            self.previou_rel_score=[0.0]*self.num_rel_cls
                    
    def get_prior_reps(self,sub_embeds,obj_embeds,union_reps,obj_infos,rel_labels=None,add_losses=dict(),rel_nums=-1, **kwargs):
        if isinstance(sub_embeds,(list,tuple)):
            sub_embeds=torch.cat(sub_embeds,dim=0)
        if isinstance(obj_embeds,(list,tuple)):
            obj_embeds=torch.cat(obj_embeds,dim=0)

        device=torch.device(f'cuda:{torch.cuda.current_device()}')
        
        predicate_proto = self.W_pred(self.rel_embed.weight)  # c = Wp x tp  i.e., semantic prototypes
        proj_predicate_proto = self.project_prot_head(self.filter_pred_prot(predicate_proto))
            
        pair_preds,pair_feats=obj_infos['pair_pred'],obj_infos['pair_feat'] # pair_feats: fused roi features, semantic features and postion features
        
        # sub_sem_reps,obj_sem_reps=self.W_obj(self.obj_embed(pair_preds[:,0].long())),self.W_obj(self.obj_embed(pair_preds[:,1].long()))
        
        sub_node_feats,obj_node_feats=pair_feats[:,0,...],pair_feats[:,1,...]
        
        cps_union_reps,cps_t_sub_reps,cps_t_obj_reps=self.cps_union_reps(union_reps),self.cps_t_sub_reps(sub_embeds),self.cps_t_obj_reps(obj_embeds)
        
        # ---------------------- generate relation edge reps from subject-object ----------------------
        # node - node ==> interaction
        if self.use_node_branch:
            edg_rel_reps=self.edge_rel_reps.expand(cps_union_reps.shape[0],-1)
            for attn_sub_node,attn_obj_node,cs_ln,cs,mlp_ln,mlp,attn_rel_pro in self.node_to_pre:
                sub_node_feats=attn_sub_node(sub_node_feats,obj_node_feats,rel_nums)
                obj_node_feats=attn_obj_node(obj_node_feats,sub_node_feats,rel_nums)
                
                entity_pairs,edg_rel_reps=torch.stack([sub_node_feats,obj_node_feats],dim=1),edg_rel_reps.unsqueeze(1)
                edg_rel_reps_out,_=cs(query=edg_rel_reps,key=entity_pairs,value=entity_pairs)
                edg_rel_reps=cs_ln(edg_rel_reps+edg_rel_reps_out)
                
                edg_rel_reps=mlp_ln(mlp(edg_rel_reps)+edg_rel_reps)

                edg_rel_reps=attn_rel_pro(edg_rel_reps.squeeze(1),proj_predicate_proto.unsqueeze(0).expand(len(rel_nums),-1,-1),rel_nums,self.num_rel_cls)

            for union_attn_entity,filter_entity,union_attn_prot,refine_edge_rel in self.refine_edge_pre:
                ln_cs,cs,ln_mlp,mlp = union_attn_entity

                entity_pairs,cps_union_reps=torch.stack([sub_node_feats,obj_node_feats],dim=1),cps_union_reps.unsqueeze(1)
                cps_union_reps_out,_ =cs(query=cps_union_reps,key=entity_pairs,value=entity_pairs)
                cps_union_reps_out=ln_cs(cps_union_reps+cps_union_reps_out)
                
                cps_union_reps_out=ln_mlp(mlp(cps_union_reps_out)+cps_union_reps_out)
                
                cps_union_reps_out,cps_union_reps=cps_union_reps_out.squeeze(1),cps_union_reps.squeeze(1)
                cps_union_reps=cps_union_reps-filter_entity(torch.cat([sub_node_feats,obj_node_feats],dim=-1))*cps_union_reps_out

                union_prot=union_attn_prot(cps_union_reps,proj_predicate_proto.unsqueeze(0).expand(len(rel_nums),-1,-1),rel_nums,self.num_rel_cls)
                
                refine_edg_rel_reps=refine_edge_rel(edg_rel_reps,union_prot,rel_nums)

            proj_edg_rel_reps=self.align_head(self.filter_noise_rel(refine_edg_rel_reps))
        else:
            proj_edg_rel_reps=None
        # ---------------------- init denoise module ----------------------
        # generate predicate reps based on triple
        cps_entity_pair_reps=self.cps_entity_pair_reps(torch.cat([cps_t_sub_reps,cps_t_obj_reps],dim=-1))
        cps_ctx_reps=cps_union_reps+cps_entity_pair_reps*cps_union_reps
        tri_rel_ctx_reps=cps_ctx_reps+cps_ctx_reps*self.gate_vis_entity(torch.cat([cps_ctx_reps,cps_entity_pair_reps],dim=-1))
        
        if self.use_denoise_branch:
            noise=torch.randn(tri_rel_ctx_reps.shape).to(device)
            noised_tri_rel_reps=tri_rel_ctx_reps+noise*self.noise_factor*tri_rel_ctx_reps
            
            for init_denoise,denoise_entity in self.denoise_modules:
                noised_tri_rel_reps=init_denoise(noised_tri_rel_reps)
                
                # ************************************************
                sub_atn_block,obj_atn_block,cps_entity_pair,rel_atn_entity,filter_entity,rel_atn_union=denoise_entity
                # attention subject features
                cps_t_sub_reps=sub_atn_block(cps_t_sub_reps,cps_t_sub_reps,rel_nums)
                
                # attention subject features
                cps_t_obj_reps=obj_atn_block(cps_t_obj_reps,cps_t_obj_reps,rel_nums)
                
                # filter subject-object features
                entity_embeds=torch.cat([cps_t_sub_reps,cps_t_obj_reps],dim=-1)
                cps_entity_embeds=cps_entity_pair(entity_embeds)
                noise_entity_out=rel_atn_entity(noised_tri_rel_reps,cps_entity_embeds,rel_nums)
                
                noised_tri_rel_reps=noised_tri_rel_reps-noise_entity_out*filter_entity(entity_embeds)
                
                # refine noised triple rel reps
                noised_tri_rel_reps=rel_atn_union(noised_tri_rel_reps,union_prot,rel_nums)
                
            denoise_tri_rel_reps=self.denoise_reps(noised_tri_rel_reps)
            
            proj_denoise_tri_rel_reps=self.align_head(self.filter_noise_rel(denoise_tri_rel_reps))
        else:
            proj_denoise_tri_rel_reps=None
        # ************ align predicate representation ************
        proj_pre_prot=self.align_head(proj_predicate_proto)
        
        if self.use_branch_fusion and self.use_denoise_branch and self.use_node_branch:
            sum_rel_reps=proj_edg_rel_reps*self.filter_tri(proj_edg_rel_reps)+proj_denoise_tri_rel_reps*self.filter_recon(proj_denoise_tri_rel_reps)
        else:
            sum_rel_reps=None
            
        # ************ using global features to refine local features ************
        if self.use_global_vis_refine:
            max_size=kwargs['enc_features'][-1].shape[-2:]
            enc_features=[F.interpolate(enc_rep,size=max_size,mode='bilinear',align_corners=False) for enc_rep in kwargs['enc_features']]  # list()
            enc_features=torch.cat(enc_features,dim=1).flatten(start_dim=2).permute(0,2,1).contiguous()
            enc_features=self.proj_glob_reps(self.filter_glob_reps(self.ds_glob_reps(enc_features)))
            
            for (refine_edg_module,refine_recon_module,refine_sum_module) in self.glob_refine_rel_reps:
                
                if self.use_node_branch:
                    proj_edg_rel_reps=refine_edg_module(proj_edg_rel_reps,kv_feats=enc_features,q_split=rel_nums)

                if self.use_denoise_branch:
                    proj_denoise_tri_rel_reps=refine_recon_module(proj_denoise_tri_rel_reps,kv_feats=enc_features,q_split=rel_nums)
                
                if self.use_branch_fusion and self.use_denoise_branch and self.use_node_branch:
                    sum_rel_reps=refine_sum_module(sum_rel_reps,kv_feats=enc_features,q_split=rel_nums)
        
        if self.use_global_rel_reps and self.use_denoise_branch and self.use_node_branch:
            # *********** merge predicate reps ***********
            glob_rel_reps=self.global_rel_reps.unsqueeze(0).expand(sum_rel_reps.shape[0],-1)
            all_rel_reps=torch.stack([sum_rel_reps,proj_denoise_tri_rel_reps,proj_edg_rel_reps],dim=1) if self.use_branch_fusion else torch.stack([proj_denoise_tri_rel_reps,proj_edg_rel_reps],dim=1)
            for merge_rel_module in self.merge_rel_reps:
                ln_sa_reps,sa_reps,ln_glob_reps_sa,glob_reps_sa,ln_ca,ca,ln_mlp,mlp=merge_rel_module
                
                all_rel_reps_attn_out,_=sa_reps(all_rel_reps,all_rel_reps,all_rel_reps)
                all_rel_reps=all_rel_reps+ln_sa_reps(all_rel_reps_attn_out)
                
                glob_rel_reps_attn_out,_=ca(glob_rel_reps.unsqueeze(1),all_rel_reps,all_rel_reps)
                glob_rel_reps=glob_rel_reps+ln_ca(glob_rel_reps_attn_out.squeeze(1))
                
                glob_rel_reps_attn_out,_=glob_reps_sa(glob_rel_reps.unsqueeze(0),glob_rel_reps.unsqueeze(0),glob_rel_reps.unsqueeze(0))
                glob_rel_reps=glob_rel_reps+ln_glob_reps_sa(glob_rel_reps_attn_out.squeeze(0))
                
                glob_rel_reps=glob_rel_reps+ln_mlp(mlp(glob_rel_reps))
        
        else:
            glob_rel_reps=None

        return (proj_edg_rel_reps,proj_denoise_tri_rel_reps,sum_rel_reps,glob_rel_reps,proj_pre_prot),add_losses        
    
    def diffusion_forward(self,condition_reps,rel_proto,rel_nums,rel_labels=None,add_losses=dict(),flexibility=0.0):
        """_summary_

        Args:
            reps (torch.tensor): shape: (b,c) init relation representation
        """
        device=condition_reps.device
        
        def diffusion_sample():
            latent_z=torch.randn_like(condition_reps).to(device)
            # z = self.flow_module(latent_z, reverse=True).view(prior_reps.shape[0], -1)
            samples = self.diff_module.sample(context=condition_reps,condition_reps=None,rel_proto=rel_proto,rel_nums=rel_nums, flexibility=flexibility)
            return samples
        
        if self.training:
            prior_reps=rel_proto[rel_labels]
            
            """
            z_m,z_v=self.enc_mean_std[0](prior_reps),self.enc_mean_std[1](prior_reps)
            latent_z=z_m+torch.exp(0.5 * z_v)*torch.randn(z_v.size(),device=z_m.device)  # reparameter
            
            w, delta_log_pw=self.flow_module(latent_z,torch.zeros([latent_z.shape[0], 1]).to(latent_z.device), reverse=False)  
            
            # calculate loss to restrict latent reps distribution 
            gs_entropy=0.5 * z_v.sum(dim=1, keepdim=False) + (0.5 * float(z_v.size(1)) * (1. + np.log(np.pi * 2)))
            
            log_pw = -0.5 * w.shape[-1] * np.log(2 * np.pi)-w.pow(2)/2
            log_pw=log_pw.view(latent_z.shape[0], -1).sum(dim=1, keepdim=True)
            log_pz = log_pw - delta_log_pw.view(latent_z.shape[0], 1)  # for flow model
            
            kl_div_loss=(-gs_entropy.mean()-log_pz.mean())*0.001
            add_losses['kl_div_loss']=add_losses.get('kl_div_loss',0.0)+kl_div_loss
            add_losses['restrict_latent_kl']=add_losses.get('restrict_latent_kl',0.0)+(-0.5 * torch.sum(1 - z_v.exp() - z_m.pow(2) + z_v))
            """
            # diffusion forward to calculate diffusion loss
            for dif_step in range(self.diff_module.num_steps):
                e_theta,e_rand,ctx_emb=self.diff_module(prior_reps,condition_reps,None,rel_proto,rel_nums,t=dif_step)    # input relation reps and reparameter latent reps  
            
                recon_loss = F.mse_loss(e_theta.view(-1, prior_reps.shape[-1]), e_rand.view(-1, prior_reps.shape[-1]), reduction='mean')    
                add_losses['diffusion_recon_loss']=add_losses.get('diffusion_recon_loss',0.0)+recon_loss
            
            return None,add_losses
               
        else:
            return diffusion_sample(),dict()
    
    def forward(self,sub_embeds,obj_embeds,union_reps,obj_infos,rel_labels=None,add_losses=dict(),proposals=None,rel_pairs=None,rel_nums=-1, **kwargs):
        device=torch.device(f'cuda:{torch.cuda.current_device()}')
        
        def cal_kl_div(mu0, logvar0, mu1=None, logvar1=None, norm_value=None):
            if mu1 is None or logvar1 is None:
                KLD = -0.5 * torch.sum(1 - logvar0.exp() - mu0.pow(2) + logvar0)
            else:
                KLD = -0.5 * (torch.sum(1 - logvar0.exp()/logvar1.exp() - (mu0-mu1).pow(2)/logvar1.exp() + logvar0 - logvar1))
            if norm_value is not None:
                KLD = KLD / float(norm_value)
            return KLD
        
        if self.step==1:
            pre_reps,add_losses=self.get_prior_reps(sub_embeds,obj_embeds,union_reps,obj_infos,rel_labels=rel_labels,add_losses=add_losses,rel_nums=rel_nums, **kwargs)
            
            edg_rel_reps,recon_tri_rel_reps,sum_rel_reps,glob_rel_reps,rel_proto=pre_reps
            
            rel_prot_norm = rel_proto / rel_proto.norm(dim=1, keepdim=True)
            
            reps_dict,predict_dict=dict(),dict()
            if self.use_branch_fusion and self.use_denoise_branch and self.use_node_branch:
                sum_rel_reps_norm=sum_rel_reps/sum_rel_reps.norm(dim=1,keepdim=True)
                sum_rel_sim=(sum_rel_reps_norm @ rel_prot_norm.t() * self.logit_scale.exp()).softmax(-1)
                
                reps_dict['sum_rel_reps']=sum_rel_reps
                predict_dict['sum_rel_sim']=sum_rel_sim
                
            if self.use_denoise_branch:
                proj_denoise_tri_rel_norm = recon_tri_rel_reps / recon_tri_rel_reps.norm(dim=1, keepdim=True)
                denoise_rel_sim=(proj_denoise_tri_rel_norm @ rel_prot_norm.t() * self.logit_scale.exp()).softmax(-1)

                reps_dict['recon_tri_rel_reps']=recon_tri_rel_reps
                predict_dict['recon_rel_sim']=denoise_rel_sim
                
            if self.use_node_branch:
                proj_edg_rel_reps_norm = edg_rel_reps / edg_rel_reps.norm(dim=1, keepdim=True)
                edg_rel_sim=(proj_edg_rel_reps_norm @ rel_prot_norm.t() * self.logit_scale.exp()).softmax(-1)  

                reps_dict['edg_rel_reps']=edg_rel_reps
                predict_dict['edg_rel_sim']=edg_rel_sim
                
            if self.use_global_rel_reps and self.use_denoise_branch and self.use_node_branch:
                glob_rel_reps_norm=glob_rel_reps/glob_rel_reps.norm(dim=1,keepdim=True)
                glob_rel_sim=(glob_rel_reps_norm @ rel_prot_norm.t() * self.logit_scale.exp()).softmax(-1)
                
                reps_dict['glob_rel_reps']=glob_rel_reps
                predict_dict['glob_rel_sim']=glob_rel_sim
                
            # *************** for step 1 ,to calculate the similar between predicate reps and prototype ***************
            
            if self.training:
                rel_labels=torch.cat(rel_labels,dim=0) if isinstance(rel_labels,(list,tuple)) else rel_labels
                add_losses=self.overall_reps_loss(rel_proto,rel_prot_norm,reps_dict,predict_dict,device,rel_labels,add_losses)
                
                if self.use_kl_modules:
                    pre_proto_mean,pre_proto_logvar=self.kl_infos[0](rel_proto),self.kl_infos[1](rel_proto)
                    pre_proto_mean,pre_proto_logvar=pre_proto_mean[rel_labels],pre_proto_logvar[rel_labels]
                    
                    for name,reps in reps_dict.items():
                        rel_reps_mean,rel_reps_logvar=self.kl_infos[0](reps),self.kl_infos[1](reps)
                        add_losses[f'{name}_kl_loss']=add_losses.get(f'{name}_kl_loss',0.0)+cal_kl_div(rel_reps_mean,rel_reps_logvar,pre_proto_mean,pre_proto_logvar)
            
            if self.use_global_rel_reps and self.use_denoise_branch and self.use_node_branch:
                pre_dist=predict_dict['glob_rel_sim']
            else:
                pre_dist=sum(predict_dict.values())
        
        else:
            with torch.no_grad():
                pre_reps,add_losses=self.get_prior_reps(sub_embeds,obj_embeds,union_reps,obj_infos,rel_labels=rel_labels,add_losses=add_losses,rel_nums=rel_nums, **kwargs)
                if self.use_glob_refine_modules:
                    recon_tri_rel_reps,edg_rel_reps,sum_rel_reps,rel_proto,glob_rel_reps=pre_reps
                    condition_reps=glob_rel_reps
                else:
                    recon_tri_rel_reps,edg_rel_reps,sum_rel_reps,rel_proto=pre_reps
                    condition_reps=torch.cat([recon_tri_rel_reps,edg_rel_reps,sum_rel_reps],dim=-1)

                sum_rel_reps_norm=sum_rel_reps/sum_rel_reps.norm(dim=1,keepdim=True)
                proj_denoise_tri_rel_norm = recon_tri_rel_reps / recon_tri_rel_reps.norm(dim=1, keepdim=True)
                proj_edg_rel_reps_norm = edg_rel_reps / edg_rel_reps.norm(dim=1, keepdim=True)
                rel_prot_norm = rel_proto / rel_proto.norm(dim=1, keepdim=True)
            
                denoise_rel_sim=(proj_denoise_tri_rel_norm @ rel_prot_norm.t() * self.logit_scale.exp()).softmax(-1)
                edg_rel_sim=(proj_edg_rel_reps_norm @ rel_prot_norm.t() * self.logit_scale.exp()).softmax(-1)            
                sum_rel_sim=(sum_rel_reps_norm @ rel_prot_norm.t() * self.logit_scale.exp()).softmax(-1)
                
                if self.use_glob_refine_modules:
                    glob_rel_reps_norm=glob_rel_reps/glob_rel_reps.norm(dim=1,keepdim=True)
                    glob_rel_sim=(glob_rel_reps_norm @ rel_prot_norm.t() * self.logit_scale.exp()).softmax(-1)
                else:
                    glob_rel_sim=0.0
                    
                if self.predict_method=="sum":
                    head_pre_dist=denoise_rel_sim+edg_rel_sim+sum_rel_sim+glob_rel_sim
                else:
                    head_pre_dist=glob_rel_sim
            
            if self.training:
                rel_labels=torch.cat(rel_labels,dim=0) if isinstance(rel_labels,(list,tuple)) else rel_labels
                _,add_losses=self.diffusion_forward(condition_reps,rel_proto,rel_nums,rel_labels,add_losses)
                
                return head_pre_dist,dict(),add_losses
            else:
                with torch.no_grad():
                    dif_recon_reps,_=self.diffusion_forward(condition_reps,rel_proto,rel_nums)  
            
            dif_recon_reps,rel_proto=dif_recon_reps.unsqueeze(dim=1).expand(-1,self.num_rel_cls,-1),rel_proto.unsqueeze(dim=0).expand(dif_recon_reps.shape[0],-1,-1)
            tail_dist=(1-((dif_recon_reps-rel_proto).norm(dim=2)**2).softmax(dim=-1))
            
            pre_dist=head_pre_dist+tail_dist
                
        torch.cuda.empty_cache()            
        return pre_dist,dict(),add_losses
    
        
    def overall_reps_loss(self,rel_proto,rel_prot_norm,extract_reps,reps_similar,device,rel_labels,add_losses):
        if self.step==1:
            add_losses=self.init_proto_loss(rel_proto,rel_prot_norm,add_losses)
        
        for name,reps in extract_reps.items():
            add_losses=self.predicate_reps_loss(reps,rel_proto,rel_labels,add_losses,loss_fun='intra_cls_loss',loss_name=f'{name}_proto_dis')
        
        # 自适应重加权损失
        self.gt_scores=self.gt_scores.to(device=device)+torch.sum(F.one_hot(rel_labels,self.num_rel_cls).to(device=device),dim=0)
        for name,reps_sim in reps_similar.items():
            if name =='glob_rel_sim':
                continue
            
            if self.use_adaptive_loss:
                with torch.no_grad():
                    pos_mask=torch.zeros(reps_sim.shape,device=device)
                    pos_mask[torch.arange(reps_sim.shape[0]),rel_labels]=1

                    pos_scores=getattr(self,f'pos_{name}_scores').to(device)+torch.sum(pos_mask.long()*reps_sim,dim=0)
                    setattr(self,f'pos_{name}_scores',pos_scores)
                    
                    neg_scores=getattr(self,f'neg_{name}_scores').to(device)+torch.sum((1-pos_mask.long())*reps_sim,dim=0)
                    setattr(self,f'neg_{name}_scores',neg_scores)
                    
                    emp_weight=self.emp_decay*getattr(self,f'{name}_emp_weight').to(device)+(1-self.emp_decay)*torch.log(1+(neg_scores/(self.gt_scores+1e-5))/(pos_scores/(self.gt_scores+1e-5)+1e-5))
                    emp_weight[0]=1e-5
                    setattr(self,f'{name}_emp_weight',emp_weight)
                    
                add_losses[f'{name}_adaptive_loss']=add_losses.get(f'{name}_adaptive_loss',0.0)+F.cross_entropy(reps_sim,rel_labels,weight=emp_weight)
            else:
                add_losses[f'{name}_loss']=add_losses.get(f'{name}_loss',0.0)+F.cross_entropy(reps_sim,rel_labels)
        return add_losses
    
    def compute_gradient_penalty(self, module, real_samples, fake_samples):
        alpha = torch.randn(real_samples.size(0), 1, 1, 1, device=real_samples.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = module(interpolates)
        fake = torch.ones(d_interpolates.shape, device=real_samples.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def init_proto_loss(self,predicate_proto,predicate_proto_norm,add_losses):
        ### Prototype Regularization  ---- cosine similarity
        target_rpredicate_proto_norm = predicate_proto_norm.clone().detach() 
        simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
        l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (self.num_rel_cls*self.num_rel_cls)  
        add_losses['l21_loss']=add_losses.get('l21_loss',0.0)+l21  # Le_sim = ||S||_{2,1}
        ### end
        
        ### Prototype Regularization  ---- Euclidean distance
        gamma2 = 7.0
        predicate_proto_a = predicate_proto.unsqueeze(dim=1).expand(-1, self.num_rel_cls, -1) 
        predicate_proto_b = predicate_proto.detach().unsqueeze(dim=0).expand(self.num_rel_cls, -1, -1)
        proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
        sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
        topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1   # obtain d-, where k2 = 1
        dist_loss = torch.max(torch.zeros(self.num_rel_cls).cuda(), -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
        add_losses['dist_loss2']=add_losses.get('dist_loss2',0.0)+dist_loss
        ### end 
        return add_losses
    
    def predicate_reps_loss(self,rel_reps,rel_center,rel_labels,add_losses,loss_fun,loss_name):
        if 'intra_cls_loss' in loss_fun:
            assert rel_labels!=None,'Please check relation labels!'
            gamma=1.0
            expand_rel_rep=rel_reps.unsqueeze(dim=1).expand(-1,self.num_rel_cls,-1) # sample_nums,rel_cls,hidden_dim
            expand_rel_center=rel_center.unsqueeze(dim=0).expand(rel_reps.shape[0],-1,-1) # sample_nums,rel_cls,hidden_dim
            
            rel_reps_dis_center=(expand_rel_rep-expand_rel_center).norm(dim=2)**2 
            neg_masks=torch.ones(rel_reps.shape[0],self.num_rel_cls,device=torch.device(f'cuda:{torch.cuda.current_device()}'))
            neg_masks[torch.arange(rel_reps.shape[0]),rel_labels]=0
            
            neg_dis=neg_masks*rel_reps_dis_center
            # sort_neg_dis,_=torch.sort(neg_dis,dim=1)
            neg_dis=neg_dis.sum(dim=1)/neg_dis.shape[0]
            
            pos_dis=rel_reps_dis_center[torch.arange(rel_reps.shape[0]),rel_labels]
            dis_loss=torch.max(torch.zeros(rel_reps.shape[0],device=torch.device(f'cuda:{torch.cuda.current_device()}')),pos_dis-neg_dis+gamma).mean()
            add_losses[loss_name]=add_losses.get(loss_name,0.0)+dis_loss
        
        if 'inter_cls_loss' in loss_fun:
            gamma=1.0
            expand_rel_rep=rel_reps.unsqueeze(dim=1).expand(-1,rel_center.shape[0],-1) # sample_nums,rel_cls,hidden_dim
            expand_rel_center=rel_center.unsqueeze(dim=0).expand(rel_reps.shape[0],-1,-1) # sample_nums,rel_cls,hidden_dim
            
            rel_reps_dis_center=(expand_rel_rep-expand_rel_center).norm(dim=2)**2 
            neg_masks=torch.ones(rel_reps.shape[0],rel_center.shape[0],device=torch.device(f'cuda:{torch.cuda.current_device()}'))
            neg_masks[torch.arange(rel_reps.shape[0]),torch.arange(rel_center.shape[0])]=0
            
            neg_dis=neg_masks*rel_reps_dis_center
            # sort_neg_dis,_=torch.sort(neg_dis,dim=1)
            neg_dis=neg_dis.sum(dim=1)/neg_dis.shape[0]
            
            pos_dis=rel_reps_dis_center[torch.arange(rel_reps.shape[0]),torch.arange(rel_center.shape[0])]
            dis_loss=torch.max(torch.zeros(rel_reps.shape[0],device=torch.device(f'cuda:{torch.cuda.current_device()}')),pos_dis-neg_dis+gamma).mean()
            add_losses[loss_name]=add_losses.get(loss_name,0.0)+dis_loss

        return add_losses

    def calculate_loss(self,relation_logits,rel_labels,proposals=None,refine_logits=None):
        # ************************ relation loss ****************************
        relation_logits,rel_labels=torch.cat(relation_logits,dim=0) if isinstance(relation_logits,(list,tuple)) else relation_logits,torch.cat(rel_labels,dim=0) if isinstance(rel_labels,(list,tuple)) else rel_labels
        rel_ce_loss=self.rel_ce_loss(relation_logits,rel_labels)
        
        rel_log_softmax = torch.log_softmax(relation_logits, dim=1)
        rel_logpt = torch.gather(rel_log_softmax, dim=1, index=rel_labels.view(-1, 1)).view(-1)
        
        rel_loss=(1-torch.exp(rel_logpt))**self.gamma*rel_ce_loss
        rel_loss=torch.mean(rel_loss)  # torch.sum(f_loss)
        
        # **************************** object loss ***************************
        if proposals is not None and refine_logits is not None:
            refine_obj_logits = cat(refine_logits, dim=0) if isinstance(refine_logits,(list,tuple)) else refine_logits
            fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            
            obj_loss = F.cross_entropy(refine_obj_logits, fg_labels.long())
        else:
            obj_loss= None
        # ********************************************************************
        
        return rel_loss,obj_loss
   
   
class DiffusionModel(nn.Module):
    """_summary_

    Args:
        nn (_type_): 基于Motifs等提取的特征进行聚类，并将不断优化聚类中心，同时引入KNN计算特征间相似度，最终基于聚类后的特征进行特征预测
    """
    def __init__(self, config, in_channels, statistics,baseline_model="PENet"):
        super().__init__()
        self.config=config
        num_head = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        dropout_rate = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        rel_layer = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER
        inner_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        
        self.baseline_model=baseline_model
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.mlp_dim = in_channels
        self.k_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM         
        self.v_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM    
        
        self.embed_dim = 300 # config.MODEL.ROI_RELATION_HEAD.PENET_EMBED_DIM
        
        obj_classes, rel_classes,fg_matrix = statistics['obj_classes'], statistics['rel_classes'],statistics['fg_matrix']
        assert self.num_rel_cls == len(rel_classes)
        self.rel_classes = rel_classes
        
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)   # load Glove for predicates
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)   # load Glove for predicates
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        self.obj_embed = nn.Embedding(len(obj_classes), self.embed_dim)
        with torch.no_grad():
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.filter_pred_prot=nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.project_prot_head = MLP(self.mlp_dim, self.mlp_dim,self.hidden_dim,2)
        
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # contrast learning
        
        # *************************** generate predicate reps based on union and entity pair reps ***************************
        self.cps_t_sub_reps,self.cps_t_obj_reps=MLP(self.hidden_dim,self.mlp_dim//2,self.hidden_dim,1),MLP(self.hidden_dim,self.mlp_dim//2,self.hidden_dim,1)

        self.cps_entity_pair_reps,self.gate_vis_entity=MLP(2*self.hidden_dim,self.mlp_dim//2,self.hidden_dim,1),MLP(2*self.hidden_dim,self.mlp_dim//2,self.hidden_dim,1)
        
        # *************************** entity node pair --> predicate reps ***************************
        self.cps_union_reps=MLP(self.pooling_dim,self.mlp_dim//2,self.hidden_dim,1)
        self.edge_rel_reps=nn.Parameter(torch.normal(mean=0, std=0.1, size=(self.hidden_dim,)))
        self.node_to_pre=nn.ModuleList([
            nn.ModuleList([
                Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate), # Enhance Node
                Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate), # Enhance Node
                nn.LayerNorm(self.hidden_dim),
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),  # generate predicate reps
                nn.LayerNorm(self.hidden_dim),
                MLP(self.hidden_dim,self.mlp_dim//2,self.hidden_dim,1),  # proj predicate reps -> predicate prototype 
                Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate) # Cross attention predicate prototype           
            ]) for _  in range(rel_layer)
        ])
        
        self.refine_edge_pre=nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.LayerNorm(self.hidden_dim),
                    nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True), 
                    nn.LayerNorm(self.hidden_dim),
                    MLP(self.hidden_dim,self.mlp_dim//2,self.hidden_dim,1), # Enhance entity weight in union features
                ]),
                nn.Sequential(
                    nn.Linear(2*self.hidden_dim,self.hidden_dim),
                    nn.Sigmoid()
                ),  # del entity features
                Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate), # Enhance predicate prototye reps in union reps
                Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate)  # Refine predicate reps
            ]) for _ in range(rel_layer)
        ])
    
        # *************************** union triple --> predicate reps ***************************
        
        self.noise_factor=nn.Parameter(torch.ones(1),requires_grad=True)  # add noise
        
        self.denoise_modules=nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.hidden_dim,self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim,self.hidden_dim),
                    nn.ReLU(inplace=True)
                ), # denoise
                nn.ModuleList([  # denoise subject/object features
                    Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate), # subject self attention
                    Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate), # object self attention
                    nn.Sequential(
                        nn.Linear(2*self.hidden_dim,self.hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(self.hidden_dim,self.hidden_dim)
                    ),
                    Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate), # predicate reps attention sub-obj reps
                    nn.Sequential(
                        nn.Linear(2*self.hidden_dim,self.hidden_dim),
                        nn.Sigmoid()
                    ),  # del entity features
                    Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate) # t_predicate reps attention union features
                ])
            ]) for _ in range(rel_layer)
        ])
        
        self.denoise_reps=nn.Sequential(
                    nn.Linear(self.hidden_dim,self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim,self.hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2)
                )
            
        # **************** Semantic consistency module ****************
        self.align_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim, 2)
        self.filter_noise_rel=nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # **************** discriminator module ****************
        self.step=config.MODEL.ROI_RELATION_HEAD.TRAIN_STEP
        if self.step!=1:
            self.build_diff_modules()
                
        self.sum_rel_emp_weight,self.pos_sum_rel_scores,self.neg_sum_rel_scores=torch.ones(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False) 
        
        # **************** process global features ****************
        self.ds_glob_reps=nn.Sequential(
            nn.Linear(5*config.MODEL.RESNETS.BACKBONE_OUT_CHANNELS,self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim,self.hidden_dim)
        )
        
        self.proj_glob_reps = MLP(self.hidden_dim, self.hidden_dim, self.mlp_dim, 2)
        self.filter_glob_reps=nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.glob_refine_rel_reps=nn.ModuleList([
            nn.ModuleList([
                Trans_block(1,num_head,self.k_dim,self.v_dim,self.mlp_dim,self.hidden_dim,dropout_rate),  # for edg rel reps
                Trans_block(1,num_head,self.k_dim,self.v_dim,self.mlp_dim,self.hidden_dim,dropout_rate),  # for denoise rel reps
            ]) for _ in range(rel_layer)
        ])
        
        self.use_glob_refine_modules=config.MODEL.ROI_RELATION_HEAD.USE_GLOB_REFINE
        if self.use_glob_refine_modules:
            self.global_rel_reps=nn.Parameter(torch.normal(mean=0, std=0.1, size=(self.mlp_dim,)))
            self.merge_rel_reps=nn.ModuleList([
                nn.ModuleList([
                    nn.LayerNorm(self.mlp_dim),
                    nn.MultiheadAttention(self.mlp_dim,num_head,dropout=dropout_rate,batch_first=True),
                    nn.LayerNorm(self.mlp_dim),
                    nn.MultiheadAttention(self.mlp_dim,num_head,dropout=dropout_rate,batch_first=True),
                    nn.LayerNorm(self.mlp_dim),
                    nn.MultiheadAttention(self.mlp_dim,num_head,dropout=dropout_rate,batch_first=True), 
                    nn.LayerNorm(self.mlp_dim),
                    MLP(self.mlp_dim,self.mlp_dim//2,self.mlp_dim,1)
                ]) for _ in range(rel_layer)
            ])
            
        self.condition_reps=nn.Parameter(torch.normal(mean=0, std=0.1, size=(self.mlp_dim,)))
        self.extract_condition_reps=nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(self.mlp_dim),
                nn.MultiheadAttention(self.mlp_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.mlp_dim),
                nn.MultiheadAttention(self.mlp_dim,num_head,dropout=dropout_rate,batch_first=True), 
                nn.LayerNorm(self.mlp_dim),
                MLP(self.mlp_dim,self.mlp_dim*2,self.mlp_dim,2)
            ]) for _ in range(rel_layer)
        ])
        
        self.dis_pre_weight,self.sim_pre_weight=nn.Parameter(torch.ones(self.num_rel_cls),requires_grad=True),nn.Parameter(torch.ones(self.num_rel_cls),requires_grad=True)
                
        self.predict_method=config.MODEL.ROI_RELATION_HEAD.PRE_RESULT
        assert self.predict_method=="sum" or ( self.predict_method==None and self.use_glob_refine_modules ), print(f'if predict method is not sum, please check using glob refine module')
        # ******************** loss ********************
        self.gamma,self.total_iters=1,config.SOLVER.MAX_ITER
        bata=0.9999
        
        per_predicate_num=np.sum(fg_matrix.numpy(),axis=(0,1))
        self.per_predicate_weight=torch.tensor([(1-bata)/(1-bata**pre_num) for pre_num in per_predicate_num],dtype=torch.float)
        self.rel_ce_loss=nn.CrossEntropyLoss(self.per_predicate_weight)
        
        self.edg_rel_emp_weight,self.tri_rel_emp_weight,self.emp_decay=torch.ones(self.num_rel_cls,requires_grad=False),torch.ones(self.num_rel_cls,requires_grad=False),0.8
        self.pos_edg_rel_scores,self.pos_tri_rel_scores,self.neg_edg_rel_scores,self.neg_tri_rel_scores,self.gt_scores=torch.zeros(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False)
    
    def build_diff_modules(self):
        """
        from .diffusion_utils import VarianceSchedule
        self.num_steps=50   # diffusion steps
        beta_1,beta_T,sched_mode=1e-4,0.02,'linear'
        self.var_sched=VarianceSchedule(self.num_steps,beta_1,beta_T,mode=sched_mode)

        num_head = self.config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        dropout_rate = self.config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        rel_layer = self.config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER

        self.time_embedding=nn.Embedding(self.num_steps,512)
        nn.init.normal_(self.time_embedding.weight, mean=0, std=1)
        
        self.gate_time_embed=nn.Sequential(
            nn.Linear(512+self.mlp_dim,self.mlp_dim),
            nn.Sigmoid()
        )

        self.diff_decoder=nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.mlp_dim,self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim,self.mlp_dim),
                    nn.ReLU(inplace=True)
                ), # denoise
                Trans_block(1,num_head,self.k_dim,self.v_dim,self.mlp_dim,self.hidden_dim,dropout_rate), # crosss attention
                Trans_block(1,num_head,self.k_dim,self.v_dim,self.mlp_dim,self.hidden_dim,dropout_rate), # self attention
                nn.Sequential(
                    nn.Linear(self.mlp_dim*2,self.mlp_dim),
                    nn.Sigmoid()
                ),
                nn.Sequential(
                    nn.Linear(self.mlp_dim,self.mlp_dim),
                    nn.ReLU(),
                    nn.Linear(self.mlp_dim,self.mlp_dim)
                )
            ]) for _ in range(rel_layer)
        ])
        
        self.fuse_denoised_reps=nn.ModuleList([
            nn.MultiheadAttention(self.mlp_dim,num_head,dropout_rate,add_bias_kv=True,batch_first=True), #self attention
            nn.MultiheadAttention(self.mlp_dim,num_head,dropout_rate,add_bias_kv=True,batch_first=True), # cross attention
            nn.Linear(self.mlp_dim,self.mlp_dim)
        ])
        """

        from .diffusion_utils import flow_model,diffusion_model
        self.num_steps=50
        
        # arg_dif_module=Argument_Diff(self.config,self.mlp_dim,num_steps=self.num_steps)
        self.flow_module=flow_model(self.mlp_dim*2,self.mlp_dim,depth=14)
        self.diff_module=diffusion_model(self.mlp_dim,arg_diff_recon=None,num_steps=self.num_steps)
        
        self.previous_res()
        self.head_weight=nn.Parameter(torch.ones(self.num_rel_cls),requires_grad=True)
        self.head_bias=nn.Parameter(torch.zeros(self.num_rel_cls),requires_grad=True)
        self.tail_weight=nn.Parameter(torch.ones(self.num_rel_cls),requires_grad=True)
        self.tail_bias=nn.Sequential(
            make_fc(self.mlp_dim,self.num_rel_cls),
            nn.Sigmoid()
        )
        
        self.fused_diff_reps=MLP(self.mlp_dim,self.hidden_dim,self.mlp_dim,2)
        self.gate_diff_reps=nn.Sequential(
            make_fc(2*self.mlp_dim,self.mlp_dim),
            nn.ReLU()
        )

        """
        from transformers import CLIPTextModel,CLIPVisionModel,AutoProcessor
        self.clip_processor=AutoProcessor.from_pretrained('/data/sdc/pretrain_ckpt/CLIP/clip-vit-base-patch32')
        self.clip_vis_model=CLIPVisionModel.from_pretrained("/data/sdc/pretrain_ckpt/CLIP/clip-vit-base-patch32")
        self.vis_embed_dim=self.clip_vis_model.config.hidden_size
        
        self.clip_token=AutoTokenizer.from_pretrained("/data/sdc/pretrain_ckpt/CLIP/clip-vit-base-patch32") 
        self.clip_text_model=CLIPTextModel.from_pretrained("/data/sdc/pretrain_ckpt/CLIP/clip-vit-base-patch32")
        self.lg_embed_dim=self.clip_text_model.config.hidden_size
        self.clip_text_model.eval()
        
        rel_des=[]
        for rel_name in self.rel_classes:
            rel_des.append(f"The relationship is {rel_name}")
        embed_rel_tokens=self.clip_token(rel_des,padding=True,return_tensors='pt')
        with torch.no_grad():
            self.embed_rel=self.clip_text_model(**embed_rel_tokens).pooler_output
        
        self.proj_clip_text=MLP(self.lg_embed_dim,self.mlp_dim,self.mlp_dim,2)
        self.proj_clip_vis=MLP(self.vis_embed_dim,self.mlp_dim,self.mlp_dim,2)
        
        num_head = self.config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        dropout_rate = self.config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        rel_layer = self.config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER
        k_dim = self.config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM         
        v_dim = self.config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM  
        
        self.proj_cond2clip=MLP(self.mlp_dim,self.mlp_dim,self.vis_embed_dim,2)
        self.refine_clip_vis_reps=nn.ModuleList([
            nn.ModuleList([
                Trans_block(1,num_head,k_dim,v_dim,self.vis_embed_dim,self.mlp_dim,dropout_rate),
                Trans_block(1,num_head,k_dim,v_dim,self.vis_embed_dim,self.mlp_dim,dropout_rate)
            ]) for _ in range(3)
        ])
        self.clip_logit_scale=nn.Parameter(torch.tensor(2.6592))
        """
        
        """
        self.refine_rel_query=nn.Parameter(torch.normal(mean=0, std=0.1, size=(self.mlp_dim,)),requires_grad=True)
        self.extract_diff_reps=nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(self.mlp_dim,num_head,dropout_rate,batch_first=True),
                nn.LayerNorm(self.mlp_dim),   # diffusion reps self attention
                nn.MultiheadAttention(self.mlp_dim,num_head,dropout_rate,batch_first=True),
                nn.LayerNorm(self.mlp_dim),   # rel_query to extract useful diffusion reps by cross attention
                nn.MultiheadAttention(self.mlp_dim,num_head,dropout_rate,batch_first=True),
                nn.LayerNorm(self.mlp_dim),   # rel query refine relation reps by cross attention
                MLP(self.mlp_dim,self.mlp_dim,self.mlp_dim,2),
                nn.LayerNorm(self.mlp_dim),
            ]) for _ in range(rel_layer)
        ])
        """

    def freeze_module(self):
        pass
        
    def get_prior_reps(self,sub_embeds,obj_embeds,union_reps,obj_infos,rel_labels=None,add_losses=dict(),rel_nums=-1, **kwargs):
        if isinstance(sub_embeds,(list,tuple)):
            sub_embeds=torch.cat(sub_embeds,dim=0)
        if isinstance(obj_embeds,(list,tuple)):
            obj_embeds=torch.cat(obj_embeds,dim=0)

        device=torch.device(f'cuda:{torch.cuda.current_device()}')
        
        predicate_proto = self.W_pred(self.rel_embed.weight)  # c = Wp x tp  i.e., semantic prototypes
        proj_predicate_proto = self.project_prot_head(self.filter_pred_prot(predicate_proto))
            
        pair_preds,pair_feats=obj_infos['pair_pred'],obj_infos['pair_feat'] # pair_feats: fused roi features, semantic features and postion features
        
        # sub_sem_reps,obj_sem_reps=self.W_obj(self.obj_embed(pair_preds[:,0].long())),self.W_obj(self.obj_embed(pair_preds[:,1].long()))
        
        sub_node_feats,obj_node_feats=pair_feats[:,0,...],pair_feats[:,1,...]
        
        cps_union_reps,cps_t_sub_reps,cps_t_obj_reps=self.cps_union_reps(union_reps),self.cps_t_sub_reps(sub_embeds),self.cps_t_obj_reps(obj_embeds)
        
        # generate predicate reps based on triple
        cps_entity_pair_reps=self.cps_entity_pair_reps(torch.cat([cps_t_sub_reps,cps_t_obj_reps],dim=-1))
        cps_ctx_reps=cps_union_reps+cps_entity_pair_reps*cps_union_reps
        tri_rel_ctx_reps=cps_ctx_reps+cps_ctx_reps*self.gate_vis_entity(torch.cat([cps_ctx_reps,cps_entity_pair_reps],dim=-1))
        
        # node - node ==> interaction
        edg_rel_reps=self.edge_rel_reps.expand(cps_union_reps.shape[0],-1)
        for attn_sub_node,attn_obj_node,cs_ln,cs,mlp_ln,mlp,attn_rel_pro in self.node_to_pre:
            sub_node_feats=attn_sub_node(sub_node_feats,obj_node_feats,rel_nums)
            obj_node_feats=attn_obj_node(obj_node_feats,sub_node_feats,rel_nums)
            
            entity_pairs,edg_rel_reps=torch.stack([sub_node_feats,obj_node_feats],dim=1),edg_rel_reps.unsqueeze(1)
            edg_rel_reps_out,_=cs(query=edg_rel_reps,key=entity_pairs,value=entity_pairs)
            edg_rel_reps=cs_ln(edg_rel_reps+edg_rel_reps_out)
            
            edg_rel_reps=mlp_ln(mlp(edg_rel_reps)+edg_rel_reps)

            edg_rel_reps=attn_rel_pro(edg_rel_reps.squeeze(1),proj_predicate_proto.unsqueeze(0).expand(len(rel_nums),-1,-1),rel_nums,self.num_rel_cls)

        for union_attn_entity,filter_entity,union_attn_prot,refine_edge_rel in self.refine_edge_pre:
            ln_cs,cs,ln_mlp,mlp = union_attn_entity

            entity_pairs,cps_union_reps=torch.stack([sub_node_feats,obj_node_feats],dim=1),cps_union_reps.unsqueeze(1)
            cps_union_reps_out,_ =cs(query=cps_union_reps,key=entity_pairs,value=entity_pairs)
            cps_union_reps_out=ln_cs(cps_union_reps+cps_union_reps_out)
            
            cps_union_reps_out=ln_mlp(mlp(cps_union_reps_out)+cps_union_reps_out)
            
            cps_union_reps_out,cps_union_reps=cps_union_reps_out.squeeze(1),cps_union_reps.squeeze(1)
            cps_union_reps=cps_union_reps-filter_entity(torch.cat([sub_node_feats,obj_node_feats],dim=-1))*cps_union_reps_out

            union_prot=union_attn_prot(cps_union_reps,proj_predicate_proto.unsqueeze(0).expand(len(rel_nums),-1,-1),rel_nums,self.num_rel_cls)
            
            refine_edg_rel_reps=refine_edge_rel(edg_rel_reps,union_prot,rel_nums)

        # **************** init denoise module ****************
        noise=torch.randn(tri_rel_ctx_reps.shape).to(device)
        noised_tri_rel_reps=tri_rel_ctx_reps+noise*self.noise_factor*tri_rel_ctx_reps
        
        for init_denoise,denoise_entity in self.denoise_modules:
            noised_tri_rel_reps=init_denoise(noised_tri_rel_reps)
            
            # ************************************************
            sub_atn_block,obj_atn_block,cps_entity_pair,rel_atn_entity,filter_entity,rel_atn_union=denoise_entity
            # attention subject features
            cps_t_sub_reps=sub_atn_block(cps_t_sub_reps,cps_t_sub_reps,rel_nums)
            
            # attention subject features
            cps_t_obj_reps=obj_atn_block(cps_t_obj_reps,cps_t_obj_reps,rel_nums)
            
            # filter subject-object features
            entity_embeds=torch.cat([cps_t_sub_reps,cps_t_obj_reps],dim=-1)
            cps_entity_embeds=cps_entity_pair(entity_embeds)
            noise_entity_out=rel_atn_entity(noised_tri_rel_reps,cps_entity_embeds,rel_nums)
            
            noised_tri_rel_reps=noised_tri_rel_reps-noise_entity_out*filter_entity(entity_embeds)
            
            # refine noised triple rel reps
            noised_tri_rel_reps=rel_atn_union(noised_tri_rel_reps,union_prot,rel_nums)
            
        denoise_tri_rel_reps=self.denoise_reps(noised_tri_rel_reps)
        
        # ************ align predicate representation ************
        proj_denoise_tri_rel_reps=self.align_head(self.filter_noise_rel(denoise_tri_rel_reps))
        proj_edg_rel_reps=self.align_head(self.filter_noise_rel(refine_edg_rel_reps))
        proj_pre_prot=self.align_head(proj_predicate_proto)
        
        # ************ using global features to refine local features ************
        max_size=kwargs['enc_features'][-1].shape[-2:]
        enc_features=[F.interpolate(enc_rep,size=max_size,mode='bilinear',align_corners=False) for enc_rep in kwargs['enc_features']]  # list()
        enc_features=torch.cat(enc_features,dim=1).flatten(start_dim=2).permute(0,2,1).contiguous()
        enc_features=self.proj_glob_reps(self.filter_glob_reps(self.ds_glob_reps(enc_features)))
        
        for (refine_edg_module,refine_recon_module) in self.glob_refine_rel_reps:

            proj_edg_rel_reps=refine_edg_module(proj_edg_rel_reps,kv_feats=enc_features,q_split=rel_nums)

            proj_denoise_tri_rel_reps=refine_recon_module(proj_denoise_tri_rel_reps,kv_feats=enc_features,q_split=rel_nums)
            
        if self.use_glob_refine_modules:
            # *********** merge predicate reps ***********
            glob_rel_reps=self.global_rel_reps.unsqueeze(0).expand(proj_denoise_tri_rel_reps.shape[0],-1)
            all_rel_reps=torch.stack([proj_denoise_tri_rel_reps,proj_edg_rel_reps],dim=1)
            for merge_rel_module in self.merge_rel_reps:
                ln_sa_reps,sa_reps,ln_glob_reps_sa,glob_reps_sa,ln_ca,ca,ln_mlp,mlp=merge_rel_module
                
                all_rel_reps_attn_out,_=sa_reps(all_rel_reps,all_rel_reps,all_rel_reps)
                all_rel_reps=all_rel_reps+ln_sa_reps(all_rel_reps_attn_out)
                
                glob_rel_reps_attn_out,_=ca(glob_rel_reps.unsqueeze(1),all_rel_reps,all_rel_reps)
                glob_rel_reps=glob_rel_reps+ln_ca(glob_rel_reps_attn_out.squeeze(1))
                
                glob_rel_reps_attn_out,_=glob_reps_sa(glob_rel_reps.unsqueeze(0),glob_rel_reps.unsqueeze(0),glob_rel_reps.unsqueeze(0))
                glob_rel_reps=glob_rel_reps+ln_glob_reps_sa(glob_rel_reps_attn_out.squeeze(0))
                
                glob_rel_reps=glob_rel_reps+ln_mlp(mlp(glob_rel_reps))
                
            return (proj_denoise_tri_rel_reps,proj_edg_rel_reps,glob_rel_reps,proj_pre_prot),add_losses
        
        return (proj_denoise_tri_rel_reps,proj_edg_rel_reps,proj_pre_prot),add_losses
    
    def diffusion_forward(self,context_reps,rel_proto,rel_nums,rel_labels=None,add_losses=dict()):
        """_summary_

        Args:
            reps (torch.tensor): shape: (b,c) init relation representation
        """
        
        """
        step_denoised=[]
        for t in range(self.num_steps):
            batch_size, reps_dim = prior_reps.size()
            if t == None:
                t = self.var_sched.uniform_sample_t(batch_size)
            else:
                t=torch.tensor([t]*batch_size,device=prior_reps.device)
            alpha_bar = self.var_sched.alpha_bars[t]
            beta = self.var_sched.betas[t]

            c0 = torch.sqrt(alpha_bar).view(-1, 1)       # (B, 1)
            c1 = torch.sqrt(1 - alpha_bar).view(-1, 1)   # (B, 1)

            e_rand = torch.randn_like(prior_reps)  # (B, d)

            noised_reps=c0 * prior_reps + c1 * e_rand

            time_emb=self.time_embedding(t)
            noised_reps=self.gate_time_embed(torch.cat([time_emb,noised_reps],dim=-1))*noised_reps+noised_reps

            for diff_decoder in self.diff_decoder:
                pre_denoise,cross_denoise,self_denoise,gate_noise,post_denoise=diff_decoder

                denoised_prior_reps=pre_denoise(noised_reps)
                denoised_prior_reps=cross_denoise(denoised_prior_reps,rel_proto.unsqueeze(0).expand(len(rel_nums),-1,-1),rel_nums)

                denoised_prior_reps=self_denoise(denoised_prior_reps,denoised_prior_reps,rel_nums)

                denoised_prior_reps=denoised_prior_reps-gate_noise(torch.cat([noised_reps,denoised_prior_reps],dim=-1))*noised_reps
                noised_reps=post_denoise(denoised_prior_reps)
        
            step_denoised.append(noised_reps)
            add_losses=self.predicate_reps_loss(noised_reps,rel_proto,rel_labels,add_losses,'intra_cls_loss','step_df_rep_proto_dist')
        
        denoised_reps=torch.stack(step_denoised,dim=1)
        denoised_self_attn,denoised_cross_attn,post_process=self.fuse_denoised_reps
        denoised_reps,_=denoised_self_attn(denoised_reps,denoised_reps,denoised_reps)
        denoised_reps,_=denoised_cross_attn(denoised_reps,rel_proto.unsqueeze(0).expand(denoised_prior_reps.shape[0],-1,-1),rel_proto.unsqueeze(0).expand(denoised_prior_reps.shape[0],-1,-1))
        denoised_reps=post_process(torch.mean(denoised_reps,dim=1))
        return denoised_reps,add_losses
        """
        
        
        device=context_reps.device
        if self.training:
            flow_input=torch.cat([context_reps,rel_proto[rel_labels]],dim=-1)
            flow_out, delta_log_pw=self.flow_module(flow_input,torch.zeros([context_reps.shape[0], 1]).to(device), reverse=False)
            
            flow_proj_reps=flow_out[:,context_reps.shape[1]:]
            
            add_losses['flow_proj_loss']=add_losses.get('flow_proj_loss',0.0)+F.mse_loss(flow_proj_reps, torch.randn_like(flow_proj_reps,device=device), reduction='mean')   
            add_losses['flow_log_loss']=add_losses.get('flow_log_loss',0.0)+delta_log_pw.mean()
        
        flow_input=torch.cat([context_reps,torch.randn_like(context_reps,device=device)],dim=-1)
        flow_out=self.flow_module(flow_input,reverse=True)
        
        flow_proj_reps=flow_out[:,context_reps.shape[1]:]
        
        if self.training:

            flow_rep_kl_div=F.kl_div(flow_proj_reps.softmax(dim=-1).log(),rel_proto.softmax(dim=-1)[rel_labels],reduction='none')
            add_losses=self.predicate_reps_loss(flow_proj_reps,rel_proto,rel_labels,add_losses,'intra_cls_loss','flow_proto_dis')
            
            kl_div_loss=torch.max(torch.zeros(flow_rep_kl_div.shape[0],device=device),flow_rep_kl_div.sum(dim=-1)).mean()
            add_losses['flow_kl_div']=add_losses.get('flow_kl_div',0.0)+kl_div_loss
            
            cosine_sim = F.cosine_similarity(flow_proj_reps, rel_proto[rel_labels], dim=-1)
            cosine_loss = 1 - cosine_sim.mean()
            
            add_losses['flow_loss']=add_losses.get('flow_loss',0.0)+cosine_loss
            
            prior_reps=rel_proto[rel_labels]
            for dif_step in range(1,self.num_steps+1):
                prior_reps,e_rand,ctx_emb=self.diff_module(prior_reps,context=flow_proj_reps,condition_reps=context_reps,rel_proto=rel_proto,rel_nums=rel_nums,t=dif_step)    # input relation reps and reparameter latent reps  
            
            recon_loss = F.mse_loss(prior_reps.view(-1, context_reps.shape[-1]), e_rand.view(-1, context_reps.shape[-1]), reduction='mean')    
            add_losses['diffusion_recon_loss']=add_losses.get('diffusion_recon_loss',0.0)+recon_loss

            samples = self.diff_module.sample(context=flow_proj_reps,condition_reps=context_reps,rel_proto=rel_proto,rel_nums=rel_nums,ret_traj=True)
            return samples,add_losses
               
        else:
            samples = self.diff_module.sample(context=flow_proj_reps,condition_reps=context_reps,rel_proto=rel_proto,rel_nums=rel_nums,ret_traj=True)
            return samples

    def embed_vis(self,img_paths,condition,rel_nums):
        read_imgs=[]
        for img_path in img_paths:
            read_imgs.append(Image.open(img_path))
        imgs=self.clip_processor(images=read_imgs,return_tensors='pt').to(device=condition.device)
        glob_vis_reps=self.clip_vis_model(**imgs).last_hidden_state
        
        condition=self.proj_cond2clip(condition)
        for cs_attn,s_attn in self.refine_clip_vis_reps:
            condition=cs_attn(condition,glob_vis_reps,q_split=rel_nums)
            condition=s_attn(condition,condition,q_split=rel_nums)
            
        vis_pooler=self.clip_vis_model.vision_model.post_layernorm(condition)
        
        proj_clip_vis=self.proj_clip_vis(vis_pooler)
        proj_clip_text=self.proj_clip_text(self.embed_rel.to(condition.device))
        
        return proj_clip_vis,proj_clip_text
        
    def forward(self,sub_embeds,obj_embeds,union_reps,obj_infos,rel_labels=None,add_losses=dict(),proposals=None,rel_pairs=None,rel_nums=-1, **kwargs):
        device=torch.device(f'cuda:{torch.cuda.current_device()}')
        
        if self.step==1:
            pre_reps,add_losses=self.get_prior_reps(sub_embeds,obj_embeds,union_reps,obj_infos,rel_labels=rel_labels,add_losses=add_losses,rel_nums=rel_nums, **kwargs)
            
            if self.use_glob_refine_modules:
                recon_tri_rel_reps,edg_rel_reps,glob_rel_reps,rel_proto=pre_reps
                pre_condition_reps=torch.stack([recon_tri_rel_reps,edg_rel_reps,glob_rel_reps],dim=1)
                reps,reps_name=[recon_tri_rel_reps,edg_rel_reps,glob_rel_reps],['recon_reps','edge_reps','glob_reps']
            else:
                recon_tri_rel_reps,edg_rel_reps,rel_proto=pre_reps
                pre_condition_reps=torch.stack([recon_tri_rel_reps,edg_rel_reps],dim=1)
                reps,reps_name=[recon_tri_rel_reps,edg_rel_reps],['recon_reps','edge_reps']
            
            condition_reps=self.condition_reps.unsqueeze(0).expand(pre_condition_reps.shape[0],-1)
            for (ln_cond_ca,cond_ca,ln_sa,sa,ln_mlp,mlp) in self.extract_condition_reps:
                
                condition_attn,_=cond_ca(condition_reps.unsqueeze(1),pre_condition_reps,pre_condition_reps)
                condition_reps=condition_reps+ln_cond_ca(condition_attn.squeeze(1))
                
                condition_attn,_=sa(condition_reps.unsqueeze(0),condition_reps.unsqueeze(0),condition_reps.unsqueeze(0))
                condition_reps=condition_reps+ln_sa(condition_attn.squeeze(0))
                
                condition_reps=condition_reps+ln_mlp(mlp(condition_reps))
             
            if self.training:
                rel_labels=torch.cat(rel_labels,dim=0) if isinstance(rel_labels,(list,tuple)) else rel_labels
                add_losses=self.construct_predicate_reps_loss(reps,reps_name,rel_proto,rel_labels,add_losses)
            
                condition_kl_div=F.kl_div(condition_reps.softmax(dim=-1).log(),rel_proto.softmax(dim=-1)[rel_labels],reduction='none')
                add_losses=self.predicate_reps_loss(condition_reps,rel_proto,rel_labels,add_losses,'intra_cls_loss','condition2proto_dist',kl_div=condition_kl_div.sum(dim=-1))
                    
                kl_div_loss=torch.max(torch.zeros(condition_kl_div.shape[0],device=torch.device(f'cuda:{torch.cuda.current_device()}')),condition_kl_div.sum(dim=-1)).mean()
                add_losses['condition_kl_div_loss']=add_losses.get('condition_kl_div_loss',0.0)+kl_div_loss
            
            sim_pre=self.sim_pre_weight*(torch.matmul(condition_reps,rel_proto.permute(1,0).contiguous()).softmax(-1))
            dif_recon_reps,rel_proto=condition_reps.unsqueeze(dim=1).expand(-1,self.num_rel_cls,-1),rel_proto.unsqueeze(dim=0).expand(condition_reps.shape[0],-1,-1)
            pre_dist=self.dis_pre_weight*(1-((dif_recon_reps-rel_proto).norm(dim=2)**2).softmax(dim=-1))+sim_pre
        
        else:
            with torch.no_grad():
                pre_reps,add_losses=self.get_prior_reps(sub_embeds,obj_embeds,union_reps,obj_infos,rel_labels=rel_labels,add_losses=add_losses,rel_nums=rel_nums, **kwargs)
            
                if self.use_glob_refine_modules:
                    recon_tri_rel_reps,edg_rel_reps,glob_rel_reps,rel_proto=pre_reps
                    pre_condition_reps=torch.stack([recon_tri_rel_reps,edg_rel_reps,glob_rel_reps],dim=1)
                    reps,reps_name=[recon_tri_rel_reps,edg_rel_reps,glob_rel_reps],['recon_reps','edge_reps','glob_reps']
                else:
                    recon_tri_rel_reps,edg_rel_reps,rel_proto=pre_reps
                    pre_condition_reps=torch.stack([recon_tri_rel_reps,edg_rel_reps],dim=1)
                    reps,reps_name=[recon_tri_rel_reps,edg_rel_reps],['recon_reps','edge_reps']
                
                condition_reps=self.condition_reps.unsqueeze(0).expand(pre_condition_reps.shape[0],-1)
                for (ln_cond_ca,cond_ca,ln_sa,sa,ln_mlp,mlp) in self.extract_condition_reps:
                    
                    condition_attn,_=cond_ca(condition_reps.unsqueeze(1),pre_condition_reps,pre_condition_reps)
                    condition_reps=condition_reps+ln_cond_ca(condition_attn.squeeze(1))
                    
                    condition_attn,_=sa(condition_reps.unsqueeze(0),condition_reps.unsqueeze(0),condition_reps.unsqueeze(0))
                    condition_reps=condition_reps+ln_sa(condition_attn.squeeze(0))
                    
                    condition_reps=condition_reps+ln_mlp(mlp(condition_reps))

                sim_pre=self.sim_pre_weight*(torch.matmul(condition_reps,rel_proto.permute(1,0).contiguous()).softmax(-1))
                expand_condition_reps,expand_rel_proto=condition_reps.unsqueeze(dim=1).expand(-1,self.num_rel_cls,-1),rel_proto.unsqueeze(dim=0).expand(condition_reps.shape[0],-1,-1)
                head_pre=self.dis_pre_weight*(1-((expand_condition_reps-expand_rel_proto).norm(dim=2)**2).softmax(dim=-1))+sim_pre
            
            if self.training:
                rel_labels=torch.cat(rel_labels,dim=0) if isinstance(rel_labels,(list,tuple)) else rel_labels
                dif_recon_reps,add_losses=self.diffusion_forward(context_reps=condition_reps,rel_proto=rel_proto,rel_nums=rel_nums,rel_labels=rel_labels,add_losses=add_losses) 
                    
                # add_losses=self.predicate_reps_loss(dif_recon_reps,rel_proto,rel_labels,add_losses,'intra_cls_loss','df_rep_proto_dist')
            else:
                dif_recon_reps=self.diffusion_forward(context_reps=condition_reps,rel_proto=rel_proto,rel_nums=rel_nums)
            
            
            if isinstance(dif_recon_reps,dict):
                assert len(dif_recon_reps)==self.diff_module.num_steps+1
                del dif_recon_reps[self.num_steps]
                dif_recon_reps_mean=self.fused_diff_reps(torch.mean(torch.stack(list(dif_recon_reps.values()),dim=1).to(device),dim=1))
                dif_recon_reps[0]=dif_recon_reps[0].to(device)
                dif_recon_reps=dif_recon_reps[0]+self.gate_diff_reps(torch.cat([dif_recon_reps[0],dif_recon_reps_mean],dim=-1))
            
            if isinstance(dif_recon_reps,dict):
                dif_recon_reps=dif_recon_reps[0].to(device)
                
            tail_bias=self.tail_bias(dif_recon_reps)
            tail_pre=torch.matmul(dif_recon_reps,rel_proto.permute(1,0).contiguous()).softmax(-1)+tail_bias
            
            if self.training:
                rel_labels=torch.cat(rel_labels,dim=0) if isinstance(rel_labels,(list,tuple)) else rel_labels
                self.previou_rel_score=torch.tensor(self.previou_rel_score,device=device)
                add_losses['tail_pre_loss']=add_losses.get('tail_pre_loss',0.0)+F.cross_entropy(tail_pre,rel_labels,weight=torch.ones_like(self.previou_rel_score,device=device)-self.previou_rel_score)
                
                
                oh_rel_labels=torch.zeros(rel_labels.shape[0],self.num_rel_cls,device=device)
                oh_rel_labels[torch.arange(rel_labels.shape[0]),rel_labels]=1
                add_losses['tail_bias_loss']=add_losses.get('tail_bias_loss',0.0)+F.l1_loss(tail_bias,oh_rel_labels-tail_pre)

        
            pre_dist=head_pre+tail_pre
            
        torch.cuda.empty_cache()            
        return pre_dist,dict(),add_losses
    
    def previous_res(self):
        logger=logging.getLogger(__name__)
        if self.step!=1:
            pre_step_res=torch.load(f'{os.path.dirname(self.config.MODEL.PRETRAINED_DETECTOR_CKPT)}/recall.pt',map_location='cpu')
            logger.info(f'load previous predicate recall score success, recall info: {pre_step_res}')
            self.previou_rel_score=[pre_step_res[rel_name] if rel_name in pre_step_res.keys() else 1.0  for rel_name in self.rel_classes]
        else:
            logger.warning('load previous recall score failed........')
            self.previou_rel_score=[0.0]*self.num_rel_cls
    
    def construct_predicate_reps_loss(self,reps,reps_name,rel_proto,rel_labels,add_losses):
        if not isinstance(reps,(list,tuple)):
            reps=[reps]
        if not isinstance(reps_name,(list,tuple)):
            reps_name=[reps_name]
        
        rel_proto_norm = rel_proto / rel_proto.norm(dim=1, keepdim=True)
        add_losses=self.init_proto_loss(rel_proto,rel_proto_norm,add_losses)
        
        def cal_kl_div(mu0, logvar0, mu1=None, logvar1=None, norm_value=None):
            if mu1 is None or logvar1 is None:
                KLD = -0.5 * torch.sum(1 - logvar0.exp() - mu0.pow(2) + logvar0)
            else:
                KLD = -0.5 * (torch.sum(1 - logvar0.exp()/logvar1.exp() - (mu0-mu1).pow(2)/logvar1.exp() + logvar0 - logvar1))
            if norm_value is not None:
                KLD = KLD / float(norm_value)
            return KLD
        
    
        for rel_rep,rel_rep_name in zip(reps,reps_name):
            add_losses=self.predicate_reps_loss(rel_rep,rel_proto,rel_labels,add_losses,loss_fun='intra_cls_loss',loss_name=f'{rel_rep_name}_proto_dis')

            kl_div=F.kl_div(rel_rep.softmax(dim=-1).log(),rel_proto.softmax(dim=-1)[rel_labels],reduction='none')
            
            rel_rep_norm=rel_rep/rel_rep.norm(dim=1,keepdim=True)
            rel_rep_sim=(rel_rep_norm@rel_proto_norm.t() * self.logit_scale.exp()).softmax(-1)
            rel_rep_ce_loss=F.cross_entropy(rel_rep_sim,rel_labels,reduction='none')
            
            add_losses['sim_ce_loss']=add_losses.get('sim_ce_loss',0.0)+torch.mean(rel_rep_ce_loss*kl_div.sum(-1))
            add_losses=self.predicate_reps_loss(rel_rep,rel_proto,rel_labels,add_losses,'intra_cls_loss','rep2proto_dist',kl_div=kl_div.sum(-1)) # for version 2
            
            kl_div_loss=torch.max(torch.zeros(kl_div.shape[0],device=torch.device(f'cuda:{torch.cuda.current_device()}')),kl_div.sum(dim=-1)).mean()
            add_losses['kl_div_loss']=add_losses.get('kl_div_loss',0.0)+kl_div_loss
            
        return add_losses
  
    def init_proto_loss(self,predicate_proto,predicate_proto_norm,add_losses):
        ### Prototype Regularization  ---- cosine similarity
        target_rpredicate_proto_norm = predicate_proto_norm.clone().detach() 
        simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
        l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (self.num_rel_cls*self.num_rel_cls)  
        add_losses['l21_loss']=add_losses.get('l21_loss',0.0)+l21  # Le_sim = ||S||_{2,1}
        ### end
        
        ### Prototype Regularization  ---- Euclidean distance
        gamma2 = 7.0
        predicate_proto_a = predicate_proto.unsqueeze(dim=1).expand(-1, self.num_rel_cls, -1) 
        predicate_proto_b = predicate_proto.detach().unsqueeze(dim=0).expand(self.num_rel_cls, -1, -1)
        proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
        sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
        topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1   # obtain d-, where k2 = 1
        dist_loss = torch.max(torch.zeros(self.num_rel_cls).cuda(), -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
        add_losses['dist_loss2']=add_losses.get('dist_loss2',0.0)+dist_loss
        ### end 
        return add_losses
    
    def predicate_reps_loss(self,rel_reps,rel_center,rel_labels,add_losses,loss_fun,loss_name,kl_div=None,extra_weight=1.0):
        if isinstance(rel_labels,(list,tuple)):
            rel_labels=torch.cat(rel_labels,dim=0)
        if 'intra_cls_loss' in loss_fun:
            assert rel_labels!=None,'Please check relation labels!'
            gamma=1.0
            expand_rel_rep=rel_reps.unsqueeze(dim=1).expand(-1,self.num_rel_cls,-1) # sample_nums,rel_cls,hidden_dim
            expand_rel_center=rel_center.unsqueeze(dim=0).expand(rel_reps.shape[0],-1,-1) # sample_nums,rel_cls,hidden_dim
            
            rel_reps_dis_center=(expand_rel_rep-expand_rel_center).norm(dim=2)**2 
            neg_masks=torch.ones(rel_reps.shape[0],self.num_rel_cls,device=torch.device(f'cuda:{torch.cuda.current_device()}'))
            neg_masks[torch.arange(rel_reps.shape[0]),rel_labels]=0
            
            neg_dis=neg_masks*rel_reps_dis_center
            # sort_neg_dis,_=torch.sort(neg_dis,dim=1)
            neg_dis=neg_dis.sum(dim=1)/neg_dis.shape[0]
            
            pos_dis=rel_reps_dis_center[torch.arange(rel_reps.shape[0]),rel_labels]
            if kl_div is None:
                kl_div=torch.ones_like(pos_dis,device=pos_dis.device)
            pos_dis=pos_dis*kl_div
            
            dis_loss=torch.max(torch.zeros(rel_reps.shape[0],device=torch.device(f'cuda:{torch.cuda.current_device()}')),pos_dis-neg_dis+gamma).mean()
            add_losses[loss_name]=add_losses.get(loss_name,0.0)+dis_loss*extra_weight
        
        if 'inter_cls_loss' in loss_fun:
            gamma=1.0
            expand_rel_rep=rel_reps.unsqueeze(dim=1).expand(-1,rel_center.shape[0],-1) # sample_nums,rel_cls,hidden_dim
            expand_rel_center=rel_center.unsqueeze(dim=0).expand(rel_reps.shape[0],-1,-1) # sample_nums,rel_cls,hidden_dim
            
            rel_reps_dis_center=(expand_rel_rep-expand_rel_center).norm(dim=2)**2 
            neg_masks=torch.ones(rel_reps.shape[0],rel_center.shape[0],device=torch.device(f'cuda:{torch.cuda.current_device()}'))
            neg_masks[torch.arange(rel_reps.shape[0]),torch.arange(rel_center.shape[0])]=0
            
            neg_dis=neg_masks*rel_reps_dis_center
            # sort_neg_dis,_=torch.sort(neg_dis,dim=1)
            neg_dis=neg_dis.sum(dim=1)/neg_dis.shape[0]
            
            pos_dis=rel_reps_dis_center[torch.arange(rel_reps.shape[0]),torch.arange(rel_center.shape[0])]
            dis_loss=torch.max(torch.zeros(rel_reps.shape[0],device=torch.device(f'cuda:{torch.cuda.current_device()}')),pos_dis-neg_dis+gamma).mean()
            add_losses[loss_name]=add_losses.get(loss_name,0.0)+dis_loss*extra_weight

        return add_losses

    def calculate_loss(self,relation_logits,rel_labels,proposals=None,refine_logits=None):
        # ************************ relation loss ****************************
        relation_logits,rel_labels=torch.cat(relation_logits,dim=0) if isinstance(relation_logits,(list,tuple)) else relation_logits,torch.cat(rel_labels,dim=0) if isinstance(rel_labels,(list,tuple)) else rel_labels
        rel_ce_loss=self.rel_ce_loss(relation_logits,rel_labels)
        
        rel_log_softmax = torch.log_softmax(relation_logits, dim=1)
        rel_logpt = torch.gather(rel_log_softmax, dim=1, index=rel_labels.view(-1, 1)).view(-1)
        
        rel_loss=(1-torch.exp(rel_logpt))**self.gamma*rel_ce_loss
        rel_loss=torch.mean(rel_loss)  # torch.sum(f_loss)
        
        # **************************** object loss ***************************
        if proposals is not None and refine_logits is not None:
            refine_obj_logits = cat(refine_logits, dim=0) if isinstance(refine_logits,(list,tuple)) else refine_logits
            fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            
            obj_loss = F.cross_entropy(refine_obj_logits, fg_labels.long())
        else:
            obj_loss= None
        # ********************************************************************
        
        return rel_loss,obj_loss


class Diffmodel(nn.Module):
    def __init__(self, config, in_channels, statistics,baseline_model="PENet"):
        super().__init__()
        from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_transformer import MultiHeadAttention,PositionwiseFeedForward
        num_head = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        dropout_rate = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        rel_layer = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER
        inner_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        
        self.config=config
        self.baseline_model=baseline_model
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.mlp_dim = in_channels
        self.k_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM         
        self.v_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM    
        
        self.embed_dim = 300 # config.MODEL.ROI_RELATION_HEAD.PENET_EMBED_DIM
        
        obj_classes, rel_classes,fg_matrix = statistics['obj_classes'], statistics['rel_classes'],statistics['fg_matrix']
        assert self.num_rel_cls == len(rel_classes)
        self.rel_classes = rel_classes
        
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)   # load Glove for predicates
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)   # load Glove for predicates
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        self.obj_embed = nn.Embedding(len(obj_classes), self.embed_dim)
        with torch.no_grad():
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.filter_pred_prot=nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.project_prot_head = MLP(self.mlp_dim, self.mlp_dim,self.hidden_dim,2)
        
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # contrast learning
        
        # *************************** generate predicate reps based on union and entity pair reps ***************************
        self.cps_t_sub_reps,self.cps_t_obj_reps=MLP(self.hidden_dim,self.mlp_dim//2,self.hidden_dim,1),MLP(self.hidden_dim,self.mlp_dim//2,self.hidden_dim,1)

        self.cps_entity_pair_reps,self.gate_vis_entity=MLP(2*self.hidden_dim,self.mlp_dim//2,self.hidden_dim,1),MLP(2*self.hidden_dim,self.mlp_dim//2,self.hidden_dim,1)
        
        # *************************** entity node pair --> predicate reps ***************************
        self.cps_union_reps=MLP(self.pooling_dim,self.mlp_dim//2,self.hidden_dim,1)
        self.edge_rel_reps=nn.Parameter(torch.normal(mean=0, std=0.1, size=(self.hidden_dim,)))
        self.node_to_pre=nn.ModuleList([
            nn.ModuleList([
                Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate), # Enhance Node
                Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate), # Enhance Node
                nn.LayerNorm(self.hidden_dim),
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),  # generate predicate reps
                nn.LayerNorm(self.hidden_dim),
                MLP(self.hidden_dim,self.mlp_dim//2,self.hidden_dim,1),  # proj predicate reps -> predicate prototype 
                Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate) # Cross attention predicate prototype           
            ]) for _  in range(rel_layer)
        ])
        
        self.refine_edge_pre=nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.LayerNorm(self.hidden_dim),
                    nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True), 
                    nn.LayerNorm(self.hidden_dim),
                    MLP(self.hidden_dim,self.mlp_dim//2,self.hidden_dim,1), # Enhance entity weight in union features
                ]),
                nn.Sequential(
                    nn.Linear(2*self.hidden_dim,self.hidden_dim),
                    nn.Sigmoid()
                ),  # del entity features
                Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate), # Enhance predicate prototye reps in union reps
                Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate)  # Refine predicate reps
            ]) for _ in range(rel_layer)
        ])
    
        # *************************** union triple --> predicate reps ***************************
        
        self.noise_factor=nn.Parameter(torch.ones(1),requires_grad=True)  # add noise
        
        self.denoise_modules=nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.hidden_dim,self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim,self.hidden_dim),
                    nn.ReLU(inplace=True)
                ), # denoise
                nn.ModuleList([  # denoise subject/object features
                    Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate), # subject self attention
                    Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate), # object self attention
                    nn.Sequential(
                        nn.Linear(2*self.hidden_dim,self.hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(self.hidden_dim,self.hidden_dim)
                    ),
                    Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate), # predicate reps attention sub-obj reps
                    nn.Sequential(
                        nn.Linear(2*self.hidden_dim,self.hidden_dim),
                        nn.Sigmoid()
                    ),  # del entity features
                    Trans_block(1,num_head,self.k_dim,self.v_dim,self.hidden_dim,self.mlp_dim,dropout_rate) # t_predicate reps attention union features
                ])
            ]) for _ in range(rel_layer)
        ])
        
        self.denoise_reps=nn.Sequential(
                    nn.Linear(self.hidden_dim,self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim,self.hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2)
                )
            
        # **************** Semantic consistency module ****************
        self.align_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim, 2)
        self.filter_noise_rel=nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # **************** discriminator module ****************
        self.step=config.MODEL.ROI_RELATION_HEAD.TRAIN_STEP
        self.build_diff_modules()
                
        self.sum_rel_emp_weight,self.pos_sum_rel_scores,self.neg_sum_rel_scores=torch.ones(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False)
        
        # **************** process global features ****************
        self.ds_glob_reps=nn.Sequential(
            nn.Linear(5*config.MODEL.RESNETS.BACKBONE_OUT_CHANNELS,self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim,self.hidden_dim)
        )
        
        self.proj_glob_reps = MLP(self.hidden_dim, self.hidden_dim, self.mlp_dim, 2)
        self.filter_glob_reps=nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.glob_refine_rel_reps=nn.ModuleList([
            nn.ModuleList([
                Trans_block(1,num_head,self.k_dim,self.v_dim,self.mlp_dim,self.hidden_dim,dropout_rate),  # for edg rel reps
                Trans_block(1,num_head,self.k_dim,self.v_dim,self.mlp_dim,self.hidden_dim,dropout_rate),  # for denoise rel reps
            ]) for _ in range(rel_layer)
        ])
        
        self.use_glob_refine_modules=config.MODEL.ROI_RELATION_HEAD.USE_GLOB_REFINE
        if self.use_glob_refine_modules:
            self.global_rel_reps=nn.Parameter(torch.normal(mean=0, std=0.1, size=(self.mlp_dim,)))
            self.merge_rel_reps=nn.ModuleList([
                nn.ModuleList([
                    nn.LayerNorm(self.mlp_dim),
                    nn.MultiheadAttention(self.mlp_dim,num_head,dropout=dropout_rate,batch_first=True),
                    nn.LayerNorm(self.mlp_dim),
                    nn.MultiheadAttention(self.mlp_dim,num_head,dropout=dropout_rate,batch_first=True),
                    nn.LayerNorm(self.mlp_dim),
                    nn.MultiheadAttention(self.mlp_dim,num_head,dropout=dropout_rate,batch_first=True), 
                    nn.LayerNorm(self.mlp_dim),
                    MLP(self.mlp_dim,self.mlp_dim*2,self.mlp_dim,2)
                ]) for _ in range(rel_layer)
            ])

        # ******************** loss ********************
        self.gamma,self.total_iters=1,config.SOLVER.MAX_ITER
        bata=0.9999
        
        per_predicate_num=np.sum(fg_matrix.numpy(),axis=(0,1))
        self.per_predicate_weight=torch.tensor([(1-bata)/(1-bata**pre_num) for pre_num in per_predicate_num],dtype=torch.float)
        self.rel_ce_loss=nn.CrossEntropyLoss(self.per_predicate_weight)
        
        self.edg_rel_emp_weight,self.tri_rel_emp_weight,self.emp_decay=torch.ones(self.num_rel_cls,requires_grad=False),torch.ones(self.num_rel_cls,requires_grad=False),0.8
        self.pos_edg_rel_scores,self.pos_tri_rel_scores,self.neg_edg_rel_scores,self.neg_tri_rel_scores,self.gt_scores=torch.zeros(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False),torch.zeros(self.num_rel_cls,requires_grad=False)
    
    def build_diff_modules(self,flow_depth=14):
        rel_layers=self.config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER
        num_head = self.config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        dropout_rate = self.config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        
        self.enc_mean_std=nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.mlp_dim,self.mlp_dim),
                nn.LayerNorm(self.mlp_dim),
                nn.ReLU(),
                nn.Linear(self.mlp_dim,self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim,self.mlp_dim)
            ),  # proj mean
            nn.Sequential(
                nn.Linear(self.mlp_dim,self.mlp_dim),
                nn.LayerNorm(self.mlp_dim),
                nn.ReLU(),
                nn.Linear(self.mlp_dim,self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim,self.mlp_dim)
            )   # proj std
        ])
        
        from .diffusion_utils import flow_model,diffusion_model
        self.flow_module=flow_model(self.mlp_dim,self.mlp_dim,flow_depth)
        self.diff_module=diffusion_model(self.mlp_dim)
        
        self.condition_reps=nn.Parameter(torch.normal(mean=0, std=0.1, size=(self.mlp_dim,)))
        self.extract_condition_reps=nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(self.mlp_dim),
                nn.MultiheadAttention(self.mlp_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.mlp_dim),
                nn.MultiheadAttention(self.mlp_dim,num_head,dropout=dropout_rate,batch_first=True), 
                nn.LayerNorm(self.mlp_dim),
                MLP(self.mlp_dim,self.mlp_dim*2,self.mlp_dim,2)
            ]) for _ in range(rel_layers)
        ])
        
        self.dis_pre_weight,self.sim_pre_weight=nn.Parameter(torch.ones(self.num_rel_cls),requires_grad=True),nn.Parameter(torch.ones(self.num_rel_cls),requires_grad=True)

        self.current_iter,self.max_iter=0,self.config.SOLVER.MAX_ITER
        
    def freeze_module(self):
        if self.step!=1:
            for name,param in self.named_parameters():
                if 'enc_mean_std' not in name and 'flow_module' not in name and 'diff_module' not in name and 'dis_pre_weight' not in name and 'sim_pre_weight' not in name:
                    param.requires_grad=False
                else:
                    param.requires_grad=True
    
    def get_prior_reps(self,sub_embeds,obj_embeds,union_reps,obj_infos,rel_labels=None,add_losses=dict(),rel_nums=-1, **kwargs):
        if isinstance(sub_embeds,(list,tuple)):
            sub_embeds=torch.cat(sub_embeds,dim=0)
        if isinstance(obj_embeds,(list,tuple)):
            obj_embeds=torch.cat(obj_embeds,dim=0)

        device=torch.device(f'cuda:{torch.cuda.current_device()}')
        
        predicate_proto = self.W_pred(self.rel_embed.weight)  # c = Wp x tp  i.e., semantic prototypes
        proj_predicate_proto = self.project_prot_head(self.filter_pred_prot(predicate_proto))
            
        pair_preds,pair_feats=obj_infos['pair_pred'],obj_infos['pair_feat'] # pair_feats: fused roi features, semantic features and postion features
        
        sub_node_feats,obj_node_feats=pair_feats[:,0,...],pair_feats[:,1,...]
        
        cps_union_reps,cps_t_sub_reps,cps_t_obj_reps=self.cps_union_reps(union_reps),self.cps_t_sub_reps(sub_embeds),self.cps_t_obj_reps(obj_embeds)
        
        # generate predicate reps based on triple
        cps_entity_pair_reps=self.cps_entity_pair_reps(torch.cat([cps_t_sub_reps,cps_t_obj_reps],dim=-1))
        cps_ctx_reps=cps_union_reps+cps_entity_pair_reps*cps_union_reps
        tri_rel_ctx_reps=cps_ctx_reps+cps_ctx_reps*self.gate_vis_entity(torch.cat([cps_ctx_reps,cps_entity_pair_reps],dim=-1))
        
        # node - node ==> interaction
        edg_rel_reps=self.edge_rel_reps.expand(cps_union_reps.shape[0],-1)
        for attn_sub_node,attn_obj_node,cs_ln,cs,mlp_ln,mlp,attn_rel_pro in self.node_to_pre:
            sub_node_feats=attn_sub_node(sub_node_feats,obj_node_feats,rel_nums)
            obj_node_feats=attn_obj_node(obj_node_feats,sub_node_feats,rel_nums)
            
            entity_pairs,edg_rel_reps=torch.stack([sub_node_feats,obj_node_feats],dim=1),edg_rel_reps.unsqueeze(1)
            edg_rel_reps_out,_=cs(query=edg_rel_reps,key=entity_pairs,value=entity_pairs)
            edg_rel_reps=cs_ln(edg_rel_reps+edg_rel_reps_out)
            
            edg_rel_reps=mlp_ln(mlp(edg_rel_reps)+edg_rel_reps)

            edg_rel_reps=attn_rel_pro(edg_rel_reps.squeeze(1),proj_predicate_proto.unsqueeze(0).expand(len(rel_nums),-1,-1),rel_nums,self.num_rel_cls)

        for union_attn_entity,filter_entity,union_attn_prot,refine_edge_rel in self.refine_edge_pre:
            ln_cs,cs,ln_mlp,mlp = union_attn_entity

            entity_pairs,cps_union_reps=torch.stack([sub_node_feats,obj_node_feats],dim=1),cps_union_reps.unsqueeze(1)
            cps_union_reps_out,_ =cs(query=cps_union_reps,key=entity_pairs,value=entity_pairs)
            cps_union_reps_out=ln_cs(cps_union_reps+cps_union_reps_out)
            
            cps_union_reps_out=ln_mlp(mlp(cps_union_reps_out)+cps_union_reps_out)
            
            cps_union_reps_out,cps_union_reps=cps_union_reps_out.squeeze(1),cps_union_reps.squeeze(1)
            cps_union_reps=cps_union_reps-filter_entity(torch.cat([sub_node_feats,obj_node_feats],dim=-1))*cps_union_reps_out

            union_prot=union_attn_prot(cps_union_reps,proj_predicate_proto.unsqueeze(0).expand(len(rel_nums),-1,-1),rel_nums,self.num_rel_cls)
            
            refine_edg_rel_reps=refine_edge_rel(edg_rel_reps,union_prot,rel_nums)

        # **************** init denoise module ****************
        noise=torch.randn(tri_rel_ctx_reps.shape).to(device)
        noised_tri_rel_reps=tri_rel_ctx_reps+noise*self.noise_factor*tri_rel_ctx_reps
        
        for init_denoise,denoise_entity in self.denoise_modules:
            noised_tri_rel_reps=init_denoise(noised_tri_rel_reps)
            
            # ************************************************
            sub_atn_block,obj_atn_block,cps_entity_pair,rel_atn_entity,filter_entity,rel_atn_union=denoise_entity
            # attention subject features
            cps_t_sub_reps=sub_atn_block(cps_t_sub_reps,cps_t_sub_reps,rel_nums)
            
            # attention subject features
            cps_t_obj_reps=obj_atn_block(cps_t_obj_reps,cps_t_obj_reps,rel_nums)
            
            # filter subject-object features
            entity_embeds=torch.cat([cps_t_sub_reps,cps_t_obj_reps],dim=-1)
            cps_entity_embeds=cps_entity_pair(entity_embeds)
            noise_entity_out=rel_atn_entity(noised_tri_rel_reps,cps_entity_embeds,rel_nums)
            
            noised_tri_rel_reps=noised_tri_rel_reps-noise_entity_out*filter_entity(entity_embeds)
            
            # refine noised triple rel reps
            noised_tri_rel_reps=rel_atn_union(noised_tri_rel_reps,union_prot,rel_nums)
            
        denoise_tri_rel_reps=self.denoise_reps(noised_tri_rel_reps)
        
        # ************ align predicate representation ************
        proj_denoise_tri_rel_reps=self.align_head(self.filter_noise_rel(denoise_tri_rel_reps))
        proj_edg_rel_reps=self.align_head(self.filter_noise_rel(refine_edg_rel_reps))
        proj_pre_prot=self.align_head(proj_predicate_proto)
        
        # ************ using global features to refine local features ************
        max_size=kwargs['enc_features'][-1].shape[-2:]
        enc_features=[F.interpolate(enc_rep,size=max_size,mode='bilinear',align_corners=False) for enc_rep in kwargs['enc_features']]  # list()
        enc_features=torch.cat(enc_features,dim=1).flatten(start_dim=2).permute(0,2,1).contiguous()
        enc_features=self.proj_glob_reps(self.filter_glob_reps(self.ds_glob_reps(enc_features)))
        
        for (refine_edg_module,refine_recon_module) in self.glob_refine_rel_reps:

            proj_edg_rel_reps=refine_edg_module(proj_edg_rel_reps,kv_feats=enc_features,q_split=rel_nums)

            proj_denoise_tri_rel_reps=refine_recon_module(proj_denoise_tri_rel_reps,kv_feats=enc_features,q_split=rel_nums)
        
        if self.use_glob_refine_modules:
            # *********** merge predicate reps ***********
            glob_rel_reps=self.global_rel_reps.unsqueeze(0).expand(proj_denoise_tri_rel_reps.shape[0],-1)
            all_rel_reps=torch.stack([proj_denoise_tri_rel_reps,proj_edg_rel_reps],dim=1)
            for merge_rel_module in self.merge_rel_reps:
                ln_sa_reps,sa_reps,ln_glob_reps_sa,glob_reps_sa,ln_ca,ca,ln_mlp,mlp=merge_rel_module
                
                all_rel_reps_attn_out,_=sa_reps(all_rel_reps,all_rel_reps,all_rel_reps)
                all_rel_reps=all_rel_reps+ln_sa_reps(all_rel_reps_attn_out)
                
                glob_rel_reps_attn_out,_=ca(glob_rel_reps.unsqueeze(1),all_rel_reps,all_rel_reps)
                glob_rel_reps=glob_rel_reps+ln_ca(glob_rel_reps_attn_out.squeeze(1))
                
                glob_rel_reps_attn_out,_=glob_reps_sa(glob_rel_reps.unsqueeze(0),glob_rel_reps.unsqueeze(0),glob_rel_reps.unsqueeze(0))
                glob_rel_reps=glob_rel_reps+ln_glob_reps_sa(glob_rel_reps_attn_out.squeeze(0))
                
                glob_rel_reps=glob_rel_reps+ln_mlp(mlp(glob_rel_reps))
                
            return (proj_denoise_tri_rel_reps,proj_edg_rel_reps,glob_rel_reps,proj_pre_prot)
        
        return (proj_denoise_tri_rel_reps,proj_edg_rel_reps,proj_pre_prot)
    
    def diffusion_forward(self,prior_reps,condition_reps,rel_proto,rel_nums,rel_labels=None,add_losses=dict(),flexibility=0.0,rtn_sample=False):
        """_summary_

        Args:
            reps (torch.tensor): shape: (b,c) init relation representation
        """
        def diffusion_sample():
            latent_z=torch.randn_like(prior_reps).to(prior_reps.device)
            z = self.flow_module(latent_z, reverse=True).view(prior_reps.shape[0], -1)
            samples = self.diff_module.sample(context=z,condition_reps=condition_reps,rel_proto=rel_proto,rel_nums=rel_nums, flexibility=flexibility)
            return samples
        
        if self.training:
            z_m,z_v=self.enc_mean_std[0](prior_reps),self.enc_mean_std[1](prior_reps)
            latent_z=z_m+torch.exp(0.5 * z_v)*torch.randn(z_v.size(),device=z_m.device)  # reparameter
            
            w, delta_log_pw=self.flow_module(latent_z,torch.zeros([latent_z.shape[0], 1]).to(latent_z.device), reverse=False)  
            
            # calculate loss to restrict latent reps distribution 
            gs_entropy=0.5 * z_v.sum(dim=1, keepdim=False) + (0.5 * float(z_v.size(1)) * (1. + np.log(np.pi * 2)))
            
            log_pw = -0.5 * w.shape[-1] * np.log(2 * np.pi)-w.pow(2)/2
            log_pw=log_pw.view(latent_z.shape[0], -1).sum(dim=1, keepdim=True)
            log_pz = log_pw - delta_log_pw.view(latent_z.shape[0], 1)  # for flow model
            
            kl_div_loss=(-gs_entropy.mean()-log_pz.mean())*0.001
            add_losses['kl_div_loss']=add_losses.get('kl_div_loss',0.0)+kl_div_loss
            add_losses['restrict_latent_kl']=add_losses.get('restrict_latent_kl',0.0)+(-0.5 * torch.sum(1 - z_v.exp() - z_m.pow(2) + z_v))
            
            # diffusion forward to calculate diffusion loss
            for dif_step in random.sample(range(1,self.diff_module.num_steps+1),min(self.diff_module.num_steps,self.diff_module.num_steps)):
                e_theta,e_rand,ctx_emb=self.diff_module(prior_reps,latent_z,condition_reps,rel_proto,rel_nums,t=dif_step)    # input relation reps and reparameter latent reps  
            
                recon_loss = F.mse_loss(e_theta.view(-1, prior_reps.shape[-1]), e_rand.view(-1, prior_reps.shape[-1]), reduction='mean')    
                add_losses['diffusion_recon_loss']=add_losses.get('diffusion_recon_loss',0.0)+recon_loss

            return None if not rtn_sample else diffusion_sample(),add_losses
               
        else:
            return diffusion_sample()
    
    def forward(self,sub_embeds,obj_embeds,union_reps,obj_infos,rel_labels=None,add_losses=dict(),proposals=None,rel_pairs=None,rel_nums=-1, **kwargs):
        device=torch.device(f'cuda:{torch.cuda.current_device()}')
        if self.training:
            self.current_iter=self.current_iter+1
        
        pre_reps=self.get_prior_reps(sub_embeds,obj_embeds,union_reps,obj_infos,rel_labels=rel_labels,add_losses=add_losses,rel_nums=rel_nums, **kwargs)
        
        if self.use_glob_refine_modules:
            recon_tri_rel_reps,edg_rel_reps,glob_rel_reps,rel_proto=pre_reps
            pre_condition_reps=torch.stack([recon_tri_rel_reps,edg_rel_reps,glob_rel_reps],dim=1)
            reps,reps_name=[recon_tri_rel_reps,edg_rel_reps,glob_rel_reps],['recon_reps','edge_reps','glob_reps']
        else:
            recon_tri_rel_reps,edg_rel_reps,rel_proto=pre_reps
            pre_condition_reps=torch.stack([recon_tri_rel_reps,edg_rel_reps],dim=1)
            reps,reps_name=[recon_tri_rel_reps,edg_rel_reps],['recon_reps','edge_reps']
        
        # ******************** for diffusion ********************
        
        condition_reps=self.condition_reps.unsqueeze(0).expand(pre_condition_reps.shape[0],-1)
        for (ln_cond_ca,cond_ca,ln_sa,sa,ln_mlp,mlp) in self.extract_condition_reps:
            
            condition_attn,_=cond_ca(condition_reps.unsqueeze(1),pre_condition_reps,pre_condition_reps)
            condition_reps=condition_reps+ln_cond_ca(condition_attn.squeeze(1))
            
            condition_attn,_=sa(condition_reps.unsqueeze(0),condition_reps.unsqueeze(0),condition_reps.unsqueeze(0))
            condition_reps=condition_reps+ln_sa(condition_attn.squeeze(0))
            
            condition_reps=condition_reps+ln_mlp(mlp(condition_reps))
                
        if self.training:
            rel_labels=torch.cat(rel_labels,dim=0) if isinstance(rel_labels,(list,tuple)) else rel_labels
            add_losses=self.construct_predicate_reps_loss(reps,reps_name,rel_proto,rel_labels,add_losses)

            if self.current_iter<self.max_iter//2:
                rtn_sample=False
                _,add_losses=self.diffusion_forward(rel_proto[rel_labels],condition_reps,rel_proto,rel_nums,rel_labels,add_losses,rtn_sample=rtn_sample) 

                dif_recon_reps=condition_reps
                
            else:
                rtn_sample=True
                dif_recon_reps,add_losses=self.diffusion_forward(rel_proto[rel_labels],condition_reps,rel_proto,rel_nums,rel_labels,add_losses,rtn_sample=rtn_sample) 
 
                dif_kl_div=F.kl_div(dif_recon_reps.softmax(dim=-1).log(),rel_proto.softmax(dim=-1)[rel_labels],reduction='none')

                add_losses=self.predicate_reps_loss(dif_recon_reps,rel_proto,rel_labels,add_losses,'intra_cls_loss','df_rep_proto_dist',kl_div=dif_kl_div.sum(dim=-1))
                
                kl_div_loss=torch.max(torch.zeros(dif_kl_div.shape[0],device=torch.device(f'cuda:{torch.cuda.current_device()}')),dif_kl_div.sum(dim=-1)).mean()
                add_losses['kl_div_loss']=add_losses.get('kl_div_loss',0.0)+kl_div_loss
            
            condition_kl_div=F.kl_div(condition_reps.softmax(dim=-1).log(),rel_proto.softmax(dim=-1)[rel_labels],reduction='none')
            add_losses=self.predicate_reps_loss(condition_kl_div,rel_proto,rel_labels,add_losses,'intra_cls_loss','condition_rep_proto_dist',kl_div=condition_kl_div.sum(dim=-1))
                
            kl_div_loss=torch.max(torch.zeros(condition_kl_div.shape[0],device=torch.device(f'cuda:{torch.cuda.current_device()}')),condition_kl_div.sum(dim=-1)).mean()
            add_losses['kl_div_loss']=add_losses.get('kl_div_loss',0.0)+kl_div_loss
            
        else:
            with torch.no_grad():
                dif_recon_reps=self.diffusion_forward(torch.randn_like(recon_tri_rel_reps).to(device),condition_reps,rel_proto,rel_nums)  
        
        sim_pre=self.sim_pre_weight*(torch.matmul(dif_recon_reps,rel_proto.permute(1,0).contiguous()).softmax(-1))
        dif_recon_reps,rel_proto=dif_recon_reps.unsqueeze(dim=1).expand(-1,self.num_rel_cls,-1),rel_proto.unsqueeze(dim=0).expand(dif_recon_reps.shape[0],-1,-1)
        pre_dist=self.dis_pre_weight*(1-((dif_recon_reps-rel_proto).norm(dim=2)**2).softmax(dim=-1))+sim_pre

        torch.cuda.empty_cache()            
        return pre_dist,dict(),add_losses
    
    def construct_predicate_reps_loss(self,reps,reps_name,rel_proto,rel_labels,add_losses):
        if not isinstance(reps,(list,tuple)):
            reps=[reps]
        if not isinstance(reps_name,(list,tuple)):
            reps_name=[reps_name]
        
        rel_proto_norm = rel_proto / rel_proto.norm(dim=1, keepdim=True)
        add_losses=self.init_proto_loss(rel_proto,rel_proto_norm,add_losses)
        
        def cal_kl_div(mu0, logvar0, mu1=None, logvar1=None, norm_value=None):
            if mu1 is None or logvar1 is None:
                KLD = -0.5 * torch.sum(1 - logvar0.exp() - mu0.pow(2) + logvar0)
            else:
                KLD = -0.5 * (torch.sum(1 - logvar0.exp()/logvar1.exp() - (mu0-mu1).pow(2)/logvar1.exp() + logvar0 - logvar1))
            if norm_value is not None:
                KLD = KLD / float(norm_value)
            return KLD
        
    
        for rel_rep,rel_rep_name in zip(reps,reps_name):
            add_losses=self.predicate_reps_loss(rel_rep,rel_proto,rel_labels,add_losses,loss_fun='intra_cls_loss',loss_name=f'{rel_rep_name}_proto_dis')

            kl_div=F.kl_div(rel_rep.softmax(dim=-1).log(),rel_proto.softmax(dim=-1)[rel_labels],reduction='none')
            
            rel_rep_norm=rel_rep/rel_rep.norm(dim=1,keepdim=True)
            rel_rep_sim=(rel_rep_norm@rel_proto_norm.t() * self.logit_scale.exp()).softmax(-1)
            rel_rep_ce_loss=F.cross_entropy(rel_rep_sim,rel_labels,reduction='none')
            
            add_losses['sim_ce_loss']=add_losses.get('sim_ce_loss',0.0)+torch.mean(rel_rep_ce_loss*kl_div.sum(-1))
            
            kl_div_loss=torch.max(torch.zeros(kl_div.shape[0],device=torch.device(f'cuda:{torch.cuda.current_device()}')),kl_div.sum(dim=-1)).mean()
            add_losses['kl_div_loss']=add_losses.get('kl_div_loss',0.0)+kl_div_loss
            
        return add_losses
    
    def predicate_reps_loss(self,rel_reps,rel_center,rel_labels,add_losses,loss_fun,loss_name,kl_div=None):
        if 'intra_cls_loss' in loss_fun:
            assert rel_labels!=None,'Please check relation labels!'
            gamma=1.0
            expand_rel_rep=rel_reps.unsqueeze(dim=1).expand(-1,self.num_rel_cls,-1) # sample_nums,rel_cls,hidden_dim
            expand_rel_center=rel_center.unsqueeze(dim=0).expand(rel_reps.shape[0],-1,-1) # sample_nums,rel_cls,hidden_dim
            
            rel_reps_dis_center=(expand_rel_rep-expand_rel_center).norm(dim=2)**2 
            neg_masks=torch.ones(rel_reps.shape[0],self.num_rel_cls,device=torch.device(f'cuda:{torch.cuda.current_device()}'))
            neg_masks[torch.arange(rel_reps.shape[0]),rel_labels]=0
            
            neg_dis=neg_masks*rel_reps_dis_center
            # sort_neg_dis,_=torch.sort(neg_dis,dim=1)
            neg_dis=neg_dis.sum(dim=1)/neg_dis.shape[0]
            
            pos_dis=rel_reps_dis_center[torch.arange(rel_reps.shape[0]),rel_labels]
            if kl_div is None:
                kl_div=torch.ones_like(pos_dis,device=pos_dis.device)
            pos_dis=pos_dis*kl_div
            
            dis_loss=torch.max(torch.zeros(rel_reps.shape[0],device=torch.device(f'cuda:{torch.cuda.current_device()}')),pos_dis-neg_dis+gamma).mean()
            add_losses[loss_name]=add_losses.get(loss_name,0.0)+dis_loss
        
        if 'inter_cls_loss' in loss_fun:
            gamma=1.0
            expand_rel_rep=rel_reps.unsqueeze(dim=1).expand(-1,rel_center.shape[0],-1) # sample_nums,rel_cls,hidden_dim
            expand_rel_center=rel_center.unsqueeze(dim=0).expand(rel_reps.shape[0],-1,-1) # sample_nums,rel_cls,hidden_dim
            
            rel_reps_dis_center=(expand_rel_rep-expand_rel_center).norm(dim=2)**2 
            neg_masks=torch.ones(rel_reps.shape[0],rel_center.shape[0],device=torch.device(f'cuda:{torch.cuda.current_device()}'))
            neg_masks[torch.arange(rel_reps.shape[0]),torch.arange(rel_center.shape[0])]=0
            
            neg_dis=neg_masks*rel_reps_dis_center
            # sort_neg_dis,_=torch.sort(neg_dis,dim=1)
            neg_dis=neg_dis.sum(dim=1)/neg_dis.shape[0]
            
            pos_dis=rel_reps_dis_center[torch.arange(rel_reps.shape[0]),torch.arange(rel_center.shape[0])]
            dis_loss=torch.max(torch.zeros(rel_reps.shape[0],device=torch.device(f'cuda:{torch.cuda.current_device()}')),pos_dis-neg_dis+gamma).mean()
            add_losses[loss_name]=add_losses.get(loss_name,0.0)+dis_loss

        return add_losses

    def init_proto_loss(self,predicate_proto,predicate_proto_norm,add_losses):
        ### Prototype Regularization  ---- cosine similarity
        target_rpredicate_proto_norm = predicate_proto_norm.clone().detach() 
        simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
        l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (self.num_rel_cls*self.num_rel_cls)  
        add_losses['l21_loss']=add_losses.get('l21_loss',0.0)+l21  # Le_sim = ||S||_{2,1}
        ### end
        
        ### Prototype Regularization  ---- Euclidean distance
        gamma2 = 7.0
        predicate_proto_a = predicate_proto.unsqueeze(dim=1).expand(-1, self.num_rel_cls, -1) 
        predicate_proto_b = predicate_proto.detach().unsqueeze(dim=0).expand(self.num_rel_cls, -1, -1)
        proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
        sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
        topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1   # obtain d-, where k2 = 1
        dist_loss = torch.max(torch.zeros(self.num_rel_cls).cuda(), -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
        add_losses['dist_loss2']=add_losses.get('dist_loss2',0.0)+dist_loss
        ### end 
        return add_losses
    
    def calculate_loss(self,relation_logits,rel_labels,proposals=None,refine_logits=None):
        # ************************ relation loss ****************************
        relation_logits,rel_labels=torch.cat(relation_logits,dim=0) if isinstance(relation_logits,(list,tuple)) else relation_logits,torch.cat(rel_labels,dim=0) if isinstance(rel_labels,(list,tuple)) else rel_labels
        rel_ce_loss=self.rel_ce_loss(relation_logits,rel_labels)
        
        rel_log_softmax = torch.log_softmax(relation_logits, dim=1)
        rel_logpt = torch.gather(rel_log_softmax, dim=1, index=rel_labels.view(-1, 1)).view(-1)
        
        rel_loss=(1-torch.exp(rel_logpt))**self.gamma*rel_ce_loss
        rel_loss=torch.mean(rel_loss)  # torch.sum(f_loss)
        
        # **************************** object loss ***************************
        if proposals is not None and refine_logits is not None:
            refine_obj_logits = cat(refine_logits, dim=0) if isinstance(refine_logits,(list,tuple)) else refine_logits
            fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            
            obj_loss = F.cross_entropy(refine_obj_logits, fg_labels.long())
        else:
            obj_loss= None
        # ********************************************************************
        
        return rel_loss,obj_loss


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


class MemoryBank(nn.Module):
    def __init__(self, max_features_per_class, in_feature_dim,rel_cls_num,save_dir,out_feature_dim=256,device=torch.device('cpu'),torch_dtype=torch.float32,with_bg=True):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.dtype,self.device=torch_dtype,device
        self.max_features_per_class = max_features_per_class
        self.sample_feature=nn.Linear(in_feature_dim,out_feature_dim)
        if with_bg:
            self.memory_bank = {cls_id:torch.tensor([],dtype=torch_dtype,device=device) for cls_id in range(rel_cls_num)}
            self.memory_num={cls_id:0 for cls_id in range(rel_cls_num)}
        else:
            self.memory_bank = {cls_id+1:torch.tensor([],dtype=torch_dtype,device=device) for cls_id in range(rel_cls_num)}
            self.memory_num={cls_id+1:0 for cls_id in range(rel_cls_num)}
        self.save_dir=save_dir
        self.load_memory=True
        
        layer_init(self.sample_feature,xavier=True)
        self.sample_feature.requires_grad_(True)
        self.sample_feature.to(device=device,dtype=torch_dtype)
        
        self.invoking=0
    
    def forward(self,class_ids, features):
        if self.load_memory:
            device=features.device
            
            if os.path.exists(f'{self.save_dir}/memory_features_{torch.cuda.current_device()}.pth'):
                self.memory_bank=torch.load(f'{self.save_dir}/memory_features_{torch.cuda.current_device()}.pth',map_location='cpu')
                for cls_id,feature in self.memory_bank.items():
                    if isinstance(feature,(list,tuple)):
                        if len(feature)==0:
                            feature=torch.tensor([])
                        else:
                            feature=torch.stack(feature,dim=0)
                    self.memory_num[cls_id]=feature.shape[0]
                    self.memory_bank[cls_id]=feature.to(device)
                self.logger.info(f'Load memory bank features success, load path: {self.save_dir}/memory_features_{torch.cuda.current_device()}.pth')
            
            self.load_memory=False
        
        detach_features=self.sample_feature(features.clone().detach())
        unique_cls_ids=class_ids.unique()

        group_features,sim_group_features,memory_group_features={},{},{}
        for cls_id in unique_cls_ids:
            cls_id_item=cls_id.item()
            cls_features=detach_features[class_ids==cls_id]
            group_features[cls_id_item]=cls_features
            
            if self.memory_num[cls_id_item]>0:
                sim_group_features[cls_id_item]=cls_features
                memory_group_features[cls_id_item]=self.memory_bank[cls_id_item]
            
            self.memory_bank[cls_id_item] = torch.cat((self.memory_bank[cls_id_item], cls_features),dim=0)[-self.max_features_per_class:,...]
            self.memory_num[cls_id_item]=self.memory_bank[cls_id_item].shape[0]
        
        self.invoking+=1
        if self.invoking%1000==0:
            torch.save(self.memory_bank,f'{self.save_dir}/memory_features_{torch.cuda.current_device()}.pth')
        
        this_group_features=[value.mean(dim=0) for value in group_features.values()]
        this_group_features=torch.stack(this_group_features,dim=0)
        this_group_features_norm=this_group_features/this_group_features.norm(dim=-1,keepdim=True)
        
        add_loss=self.calculate_semantic_loss(this_group_features,this_group_features_norm)
        
        assert len(memory_group_features)==len(sim_group_features)
        if len(memory_group_features)==0:
            return add_loss
        else:
            memory_group_features=[value.mean(dim=0) for value in memory_group_features.values()]
            sim_group_features=[value.mean(dim=0) for value in sim_group_features.values()]
        # ************* Similarity loss between features stored in the memory bank *************
        memory_group_features=torch.stack(memory_group_features,dim=0).detach()
        sim_group_features=torch.stack(sim_group_features,dim=0)
        
        add_loss.update(self.calculate_similar_loss(memory_group_features,sim_group_features,torch.arange(memory_group_features.shape[0])))
        
        rel_rep_sim_matrix=F.normalize(sim_group_features,dim=-1)@memory_group_features.t().contiguous()
        
        pos_samples = rel_rep_sim_matrix[torch.arange(rel_rep_sim_matrix.shape[0]), torch.arange(memory_group_features.shape[0])].view(rel_rep_sim_matrix.shape[0],-1)
        
        neg_matrix=torch.ones_like(rel_rep_sim_matrix,dtype=torch.long)
        neg_matrix[torch.arange(rel_rep_sim_matrix.shape[0]), torch.arange(memory_group_features.shape[0])] = 0
        neg_samples=rel_rep_sim_matrix[neg_matrix].view(rel_rep_sim_matrix.shape[0],-1)
        
        add_loss['memory_sim_loss']=add_loss.get('memory_sim_loss',0.0)+torch.mean((1 - pos_samples) ** 2)+torch.mean(F.relu(neg_samples - 0.1) ** 2)
        
        return add_loss
    
    def calculate_semantic_loss(self,semantic_feature,semantic_feature_norm):
        add_losses=dict()
        
        ### Prototype Regularization  ---- cosine similarity
        target_rpredicate_proto_norm = semantic_feature_norm.clone().detach() 
        simil_mat = semantic_feature_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
        l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (51*51)  
        add_losses.update({"memory_l21_loss": l21})  # Le_sim = ||S||_{2,1}
        ### end
        
        ### Prototype Regularization  ---- Euclidean distance
        gamma2 = 7.0
        predicate_proto_a = semantic_feature.unsqueeze(dim=1).expand(-1, semantic_feature.shape[0], -1) 
        predicate_proto_b = semantic_feature.detach().unsqueeze(dim=0).expand(semantic_feature.shape[0], -1, -1)
        proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
        sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
        topK_proto_dis = sorted_proto_dis_mat[:, :11].sum(dim=1) / 10   # obtain d-, where k2 = 1
        dist_loss = torch.max(torch.zeros(semantic_feature.shape[0]).cuda(), -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
        add_losses.update({"memory_dist_loss2": dist_loss})
        ### end
         
        return add_losses
        
    def calculate_similar_loss(self,semantic_feature,rel_rep,rel_labels,loss_name="memory_loss_dis"):
        add_losses=dict()
        ###  Prototype-based Learning  ---- Euclidean distance
        # rel_labels = cat(rel_labels, dim=0) if isinstance(rel_labels,(list,tuple)) else rel_labels
        gamma1 = 1.0
        rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, semantic_feature.shape[0], -1)  # r
        predicate_proto_expand = semantic_feature.unsqueeze(dim=0).expand(rel_rep.size(0), -1, -1)  # ci
        distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2    # Distance Set G, gi = ||r-ci||_2^2
        mask_neg = torch.ones(rel_rep.size(0), semantic_feature.shape[0]).cuda()  
        mask_neg[torch.arange(rel_rep.size(0)), rel_labels] = 0
        distance_set_neg = distance_set * mask_neg
        distance_set_pos = distance_set[torch.arange(rel_rep.size(0)), rel_labels]  # gt i.e., g+
        sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
        topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10  # obtaining g-, where k1 = 10, 
        loss_sum = torch.max(torch.zeros(rel_rep.size(0)).cuda(), distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
        add_losses.update({loss_name: loss_sum})     # Le_euc = max(0, (g+) - (g-) + gamma1)
        ### end 
        
        return add_losses
    
    def predict_similarity(self, rel_rep):
        if min(self.memory_num.values())==0:
            return None
        
        all_memory_features=[value.mean(dim=0) for value in self.memory_bank.values()]
        all_memory_features=torch.stack(all_memory_features,dim=0).detach()
        all_memory_features_norm=all_memory_features/all_memory_features.norm(dim=-1,keepdim=True)
        
        detach_rel_rep=self.sample_feature(rel_rep.clone().detach())
        rel_rep_sim_matrix=F.normalize(detach_rel_rep,dim=-1)@all_memory_features_norm.t().contiguous()
        
        return rel_rep_sim_matrix
    
