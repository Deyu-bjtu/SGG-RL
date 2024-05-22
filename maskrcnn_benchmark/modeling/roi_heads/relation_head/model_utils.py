import copy
import glob
import json
import math
import os
import random
import re
import time
import PIL
from PIL import Image
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from maskrcnn_benchmark.data.datasets.visual_genome import load_info
from maskrcnn_benchmark.modeling.roi_heads.relation_head.llava_llama import LlavaLlamaForCausalLM
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_motifs import FrequencyBias
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_transformer import MultiHeadAttention, PositionwiseFeedForward
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_vctree import VCTreeLSTMContext
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_relation import layer_init
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.utils.comm import all_gather_with_grad, concat_all_gather, get_rank,find_linear_layers
from .utils_motifs import rel_vectors, obj_edge_vectors, to_onehot, nms_overlaps, encode_box_info 
from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.modeling.roi_heads.relation_head.conversation import conv_templates
from maskrcnn_benchmark.modeling.roi_heads.relation_head.llava_arch import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX, UNION_IMAGE_INDEX,UNION_IMAGE_TOKEN
import transformers
import logging

class Base_LLM(nn.Module):
    def __init__(self,logger) -> None:
        super().__init__()
        """
            precision: default: 'bf16'  choices: "fp32", "bf16", "fp16"
        """
        vision_tower="/data/sdb/pretrain_ckpt/vit-l14"
        llm_version='/data/sdb/pretrain_ckpt/llava-llama-2-7b'
        self.precision='bf16'
        lora_enable=True
        lora_r=64
        lora_alpha=16
        lora_dropout=0.05
        lora_target_modules=["q_proj", "v_proj"]  # 'k_proj','o_proj'
     
        if self.precision == "bf16":
            self.torch_dtype = torch.bfloat16
        elif self.precision == "fp16":
            self.torch_dtype = torch.half

        self.device=torch.device(f'cuda:{torch.cuda.current_device()}')

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            llm_version,
            cache_dir=None,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        
        logger.info('LLAVA Model device: {}  token id nums: {} torch dtype: {}'.format(self.device,len(self.tokenizer),self.torch_dtype))
        
        config=transformers.AutoConfig.from_pretrained(llm_version)
        if vision_tower is not None:
            config.mm_vision_tower=vision_tower
            
        self.lm = LlavaLlamaForCausalLM.from_pretrained(
            llm_version,
            torch_dtype=self.torch_dtype,  # torch.float32,torch.half,torch.float16
            cache_dir=None,
            config=config
        )
        
        self.lm.config.eos_token_id = self.tokenizer.eos_token_id
        self.lm.config.bos_token_id = self.tokenizer.bos_token_id
        self.lm.config.pad_token_id = self.tokenizer.pad_token_id
        self.lm.config.use_cache = False
        
        self.lm.requires_grad_(False)
  
        if lora_enable and lora_r>0:
            from peft import LoraConfig, get_peft_model,PeftModel
            lora_target_modules = find_linear_layers(
                self.lm, lora_target_modules
            )
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.lm = get_peft_model(self.lm, lora_config)
            self.lm.print_trainable_parameters()
            self.save_pretrained=self.lm.save_pretrained

        self.lm.config.tune_mm_mlp_adapter = True
        self.lm.config.freeze_mm_mlp_adapter = False
        self.lm.config.mm_use_im_start_end = False
        self.lm.config.mm_use_im_patch_token = False
        self.lm.config.sep_image_conv_front = False
        self.lm.config.pretrain_mm_mlp_adapter=f"{llm_version}/mm_projector.bin"
        
        if vision_tower is not None:
            self.lm.get_model().initialize_vision_modules(
                vision_tower=vision_tower,
                mm_vision_select_layer=-2,
                mm_vision_select_feature='patch',
                pretrain_mm_mlp_adapter=f"{llm_version}/mm_projector.bin",
                use_zero3=False
            )
        
        vision_tower = self.lm.get_vision_tower()
        vision_tower.to(dtype=self.torch_dtype, device=self.device)
        
        self.vision_processor = vision_tower.image_processor
    
    def add_token(self,rel_classes,external_tokens=[],logger=None):
        add_token_nums=0
        
        self.rel_map=dict()
        for rel_name in rel_classes:
            if len(self.tokenizer(rel_name).input_ids)>2:
                add_token_nums+=self.tokenizer.add_tokens(rel_name)

            token_ids=self.tokenizer(rel_name).input_ids
            assert len(token_ids)==2 or self.tokenizer.decode(token_ids[-1])==rel_name
            self.rel_map[rel_name]=token_ids[-1]
        
        self.all_label_token_ids=list(self.rel_map.values())
        logger.info(f'Relation map dict: {self.rel_map}')

        self.all_label_token_ids=torch.tensor(self.all_label_token_ids,dtype=torch.long,device=self.device)
        
        self.special_token_map=dict()
        for external_token in external_tokens:
            add_token_nums+= self.tokenizer.add_tokens(external_token,special_tokens=True)
            self.special_token_map[external_token]=self.tokenizer(external_token, add_special_tokens=False).input_ids[-1]  
            
        logger.info(f'Add token success, add token number: {add_token_nums}, external token id maps: {self.special_token_map}')
        
        return add_token_nums
    
    def init_tokenizer_weight(self,num_new_tokens,logger=None):
        if num_new_tokens > 0:
            
            ori_input_embeddings=self.lm.get_input_embeddings().weight.data
            ori_output_embeddings=self.lm.get_output_embeddings().weight.data
            
            self.lm.resize_token_embeddings(len(self.tokenizer))
            
            input_embeddings = self.lm.get_input_embeddings().weight.data
            output_embeddings = self.lm.get_output_embeddings().weight.data
            
            input_embeddings[:-num_new_tokens]=ori_input_embeddings
            output_embeddings[:-num_new_tokens]=ori_output_embeddings
            
            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            
            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            logger.info(f'The original token embedding weight shape: {ori_input_embeddings.shape}, the original lm head weight shape: {ori_output_embeddings.shape}, the new token embedding weight shape: {input_embeddings.shape}, the new lm head weight shape: {output_embeddings.shape}\nthe alignment status of the original and new token embedding: {torch.equal(self.lm.get_input_embeddings().weight.data[:-num_new_tokens],ori_input_embeddings)}, the alignment of the original and new lm head state:  {torch.equal(self.lm.get_output_embeddings().weight.data[:-num_new_tokens],ori_output_embeddings)}')


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


class LVM4SGG(nn.Module):
    # 关系细化与v3一致，只修改对象预测方法
    def __init__(self, config, in_channels):
        super(LVM4SGG, self).__init__()

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
        self.obj_embed = nn.Embedding(self.num_obj_cls, embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
        
        ##### refine image/text features
        pretrain_clip_model,llm_version,dict_file='/data/sdb/pretrain_ckpt/CLIP/clip-vit-base-patch32','/data/sdb/pretrain_ckpt/LLAMA/llama-2-7b-hf',"/data/sda/SGG_data/VG/VG-SGG-dicts-with-attri.json"
        self.caption_base_path='/data/sda/SGG_data/VG/LLAVA_captions'
        self.ind_to_classes, self.ind_to_predicates, self.ind_to_attributes = load_info(dict_file) # contiguous 151, 51 containing __background__
        
        self.clip_processor=transformers.AutoProcessor.from_pretrained(pretrain_clip_model)
        self.clip_vision_model=transformers.CLIPVisionModel.from_pretrained(pretrain_clip_model)
        # self.clip_vision_model.eval()
        
        self.lg_tokenizer=transformers.AutoTokenizer.from_pretrained(
            llm_version,
            cache_dir=None,
            padding_side="right",
            use_fast=False,
        )
        self.lg_tokenizer.pad_token = self.lg_tokenizer.unk_token
        
        llama_cfg=transformers.AutoConfig.from_pretrained(llm_version)
        
        load_llm_embed_ckpt=torch.load(f'{llm_version}/pytorch_model-00001-of-00002.bin')['model.embed_tokens.weight']
            
        self.lg_embed=nn.Embedding(llama_cfg.vocab_size, llama_cfg.hidden_size, llama_cfg.pad_token_id)
        self.lg_embed.weight.data.copy_(load_llm_embed_ckpt)
        self.lg_embed.eval()
        
        # init semantic infomations
        # with torch.no_grad():
        #     self.s_pred_tokens=self.lg_tokenizer(text=self.ind_to_predicates,padding=True,return_tensors="pt")
        #     self.s_pred_reps=self.lg_embed(self.s_pred_tokens.input_ids[:,1:])
        
        # map clip vision features to align FasterRCNN ROI features
        self.align_roi=make_fc(self.clip_vision_model.config.hidden_size, self.hidden_dim) 
        
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])
        
        ##### refine object labels
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_cls) 
        self.lin_obj_cyx = make_fc(in_channels + embed_dim + 128, self.hidden_dim)
  
        ##### refine predicate spatial labels
        self.p_pos=make_fc(128,self.hidden_dim)
        self.p_entity=make_fc(in_channels,self.hidden_dim*2)
        
        # ******************************* vision refine modules *******************************
        self.sample_union_rep=MLP(in_channels,self.hidden_dim,self.hidden_dim,2)
        self.union_refine_global=nn.ModuleList([
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
        self.global_refine_roi=nn.ModuleList([
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
        
        # ******************************* feature space align *******************************
        
        # project embed semantic features (all semantic information using clip)
        self.p_prompt=MLP(llama_cfg.hidden_size, self.hidden_dim // 2, self.hidden_dim, 2)
        self.p_sub = MLP(llama_cfg.hidden_size, self.hidden_dim // 2, self.hidden_dim, 2)
        self.p_obj = MLP(llama_cfg.hidden_size, self.hidden_dim // 2, self.hidden_dim, 2)
        # self.p_pred = MLP(llama_cfg.hidden_size, self.hidden_dim // 2, self.hidden_dim, 2)
        
        # project all vision features to semantic space
        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.hidden_dim, self.hidden_dim*2), nn.ReLU(True),
            nn.Dropout(dropout_rate), nn.Linear(self.hidden_dim*2, self.hidden_dim)
        ])
        
        
        # ******************************* Refine Semantic features *******************************
        self.gate_sub=make_fc(self.hidden_dim*2,self.hidden_dim)
        self.gate_obj=make_fc(self.hidden_dim*2,self.hidden_dim)
        
        self.rel_query=nn.Parameter(torch.normal(mean=0, std=0.1, size=(self.hidden_dim,)))
        
        self.query_init=nn.ModuleList([
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
        self.union_refine_query=nn.ModuleList([
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
        self.prompt_refine_query=nn.ModuleList([
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
        
        self.query_pre=make_fc(self.hidden_dim,self.num_rel_cls)
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
        with torch.no_grad():
            s_obj_tokens=self.lg_tokenizer(text=[self.ind_to_classes[i] for i in entity_preds],padding=True,return_tensors="pt").to(current_device)
            entity_sem_reps=self.lg_embed(s_obj_tokens.input_ids[:,1:]).split(num_objs,dim=0)
        
        pos_embeds=pos_embeds.split(num_objs,dim=0)
        union_features=union_features.split(num_rels,dim=0)
        
        union_vis_reps,sub_sem_reps,obj_sem_reps,caption_reps,glob_sem_reps=[],[],[],[],[]
        for batch_idx,(proposal,sub_vis_rep,obj_vis_rep,entity_sem_rep,rel_pair_idx,pos_embed,union_feature) in enumerate(zip(proposals,sub_vis_reps,obj_vis_reps,entity_sem_reps,rel_pair_idxs,pos_embeds,union_features)):
            image = Image.open(proposal.get_field('file_name'))
            image_inputs = self.clip_processor(images=image, return_tensors="pt").to(current_device)
            img_encode_out=self.clip_vision_model(**image_inputs)
            img_rep = self.align_roi(img_encode_out.last_hidden_state[:,1:,:])  # without cls token

            sub_pos_embed,obj_pos_embed=self.p_pos(pos_embed[rel_pair_idx[:,0]]),self.p_pos(pos_embed[rel_pair_idx[:,1]])
            sub_vis_rep,obj_vis_rep=sub_vis_rep[rel_pair_idx[:,0]],obj_vis_rep[rel_pair_idx[:,1]]
            
            # ********************************************* refine vision roi features ***************************************************
            sub_geo_rep,obj_geo_rep=sub_vis_rep+F.relu(sub_pos_embed),obj_vis_rep+F.relu(obj_pos_embed)
            union_vis_rep,expand_img_rep=self.sample_union_rep(union_feature).unsqueeze(1),img_rep.expand(sub_geo_rep.shape[0],-1,-1)
            
            for (s_attn,s_norm,c_attn,c_norm,ffn,ffn_norm) in self.union_refine_global:
                attn_output, _ =s_attn(query=expand_img_rep,key=expand_img_rep,value=expand_img_rep)
                expand_img_rep=s_norm(expand_img_rep+attn_output)
                
                attn_output, _ =c_attn(query=expand_img_rep,key=union_vis_rep,value=union_vis_rep)
                expand_img_rep=c_norm(expand_img_rep+attn_output)
                
                expand_img_rep=ffn_norm(ffn(expand_img_rep)+expand_img_rep)
            
            sub_geo_rep,obj_geo_rep=sub_geo_rep.unsqueeze(1),obj_geo_rep.unsqueeze(1)
            for (c_attn,c_norm,ffn,ffn_norm) in self.global_refine_roi:                
                attn_output, _ =c_attn(query=sub_geo_rep,key=expand_img_rep,value=expand_img_rep)
                sub_geo_rep=c_norm(sub_geo_rep+attn_output)
                
                sub_geo_rep=ffn_norm(ffn(sub_geo_rep)+sub_geo_rep)
                # ******************************************************************************************************************
                attn_output, _ =c_attn(query=obj_geo_rep,key=expand_img_rep,value=expand_img_rep)
                obj_geo_rep=c_norm(obj_geo_rep+attn_output)
                
                obj_geo_rep=ffn_norm(ffn(obj_geo_rep)+obj_geo_rep)
                
            # ********************************************* refine semantic features ***************************************************
            # refine object semantic features
            sub_sem_rep,obj_sem_rep=entity_sem_rep[rel_pair_idx[:,0]],entity_sem_rep[rel_pair_idx[:,1]]
            sub_sem_rep,obj_sem_rep=self.p_sub(sub_sem_rep),self.p_obj(obj_sem_rep)
            vis2sem_sub,vis2sem_obj=self.vis2sem(sub_geo_rep).expand(-1,sub_sem_rep.shape[1],-1),self.vis2sem(obj_geo_rep).expand(-1,obj_sem_rep.shape[1],-1)
            
            gate_sub=F.sigmoid(self.gate_sub(torch.cat([sub_sem_rep,vis2sem_sub],dim=-1)))
            gate_obj=F.sigmoid(self.gate_obj(torch.cat([obj_sem_rep,vis2sem_obj],dim=-1)))
            
            sub_sem_rep,obj_sem_rep=sub_sem_rep+vis2sem_sub*gate_sub,obj_sem_rep+vis2sem_obj*gate_obj
            sub_sem_reps.append(sub_sem_rep)
            obj_sem_reps.append(obj_sem_rep)
            union_vis_reps.append(union_vis_rep)
            
            glob_sem_reps.append(self.vis2sem(expand_img_rep))
            
            caption_path=f'{self.caption_base_path}/{os.path.basename(proposal.get_field("file_name")).split(".")[0]}.json'
            with open(caption_path,'r') as cap_file:
                with torch.no_grad():
                    try:
                        cap_tokens=self.lg_tokenizer(text=json.load(cap_file)['caption'],padding=True,return_tensors="pt").to(current_device)
                    except:
                        raise ValueError('Error file: {caption_path}')
                    caption_reps.append(self.p_prompt(self.lg_embed(cap_tokens.input_ids[:,1:]).expand(sub_sem_rep.shape[0],-1,-1)))
        
        # refine predicate semantic features
        sub_max_token_len,obj_max_token_len,cap_max_token_len=max([i.shape[1] for i in sub_sem_reps]),max([i.shape[1] for i in obj_sem_reps]),max([i.shape[1] for i in caption_reps])
        sub_sem_reps,obj_sem_reps,union_vis_reps=torch.cat([F.pad(ten,(0,0,0,sub_max_token_len-ten.shape[1],0,0)) for ten in sub_sem_reps],dim=0),torch.cat([F.pad(ten,(0,0,0,sub_max_token_len-ten.shape[1],0,0)) for ten in obj_sem_reps],dim=0),torch.cat(union_vis_reps,dim=0)
        rel_query=self.rel_query.expand(sub_sem_reps.shape[0],1,-1)
        
        union_entity_reps,union_sem_reps,caption_reps=torch.cat([sub_sem_reps,obj_sem_reps],dim=1),self.vis2sem(union_vis_reps),torch.cat([F.pad(ten,(0,0,0,cap_max_token_len-ten.shape[1],0,0)) for ten in caption_reps],dim=0)
        
        for (s_attn,s_norm,c_attn,c_norm,ffn,ffn_norm) in self.query_init:
            s_attn_output, _= s_attn(query=rel_query,key=rel_query,value=rel_query)
            rel_query=s_norm(rel_query+s_attn_output)
            
            attn_output, _ =c_attn(query=rel_query,key=union_entity_reps,value=union_entity_reps)
            rel_query=c_norm(rel_query+attn_output)
            
            rel_query=ffn_norm(ffn(rel_query)+rel_query)
        
        for (s_attn,s_norm,c_attn,c_norm,ffn,ffn_norm) in self.union_refine_query:
            s_attn_output, _= s_attn(query=rel_query,key=rel_query,value=rel_query)
            rel_query=s_norm(rel_query+s_attn_output)
            
            attn_output, _ =c_attn(query=rel_query,key=union_sem_reps,value=union_sem_reps)
            rel_query=c_norm(rel_query+attn_output)
            
            rel_query=ffn_norm(ffn(rel_query)+rel_query)
        
        # using image caption to refine query
        tri_sem_reps=torch.cat([sub_sem_reps,rel_query,obj_sem_reps],dim=1)
        for (s_attn,s_norm,c_attn,c_norm,ffn,ffn_norm) in self.prompt_refine_query:
            s_attn_output, _= s_attn(query=tri_sem_reps,key=tri_sem_reps,value=tri_sem_reps)
            tri_sem_reps=s_norm(tri_sem_reps+s_attn_output)
            
            attn_output, _ =c_attn(query=tri_sem_reps,key=caption_reps,value=caption_reps)
            tri_sem_reps=c_norm(tri_sem_reps+attn_output)
            
            tri_sem_reps=ffn_norm(ffn(tri_sem_reps)+tri_sem_reps)
        
        assert tri_sem_reps.shape[1]==sub_max_token_len+obj_max_token_len+1
        rel_query=tri_sem_reps[:,sub_max_token_len+1,...]
        
        rel_dists=self.query_pre(rel_query)
        
        if self.training:
            rel_labels=torch.cat(rel_labels,dim=0)
            
            glob_sem_reps=torch.cat(glob_sem_reps,dim=0)
            sim_loss=F.cosine_similarity(torch.mean(tri_sem_reps,dim=1),torch.mean(glob_sem_reps,dim=1),dim=-1)
            add_losses['sim_loss']=1-sim_loss.mean()
            
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
    

class PE_V2(nn.Module):
    def __init__(self, config, in_channels):
        super(PE_V2, self).__init__()

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
    
    
class Qformer(nn.Module):
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
        
        self.bert_encoder.encoder.eval()
        for name,param in self.bert_encoder.encoder.named_parameters():
            param.requires_grad_(False)
        
        self.img_cls=nn.Parameter(torch.randn(roi_dim))
        self.img_proj = nn.Sequential(
            nn.Linear(roi_dim, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.bert_cfg.hidden_size)
        )
        
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
            
            extended_attention_mask=torch.cat([extended_attention_mask[:,:,:,:3],extended_attention_mask],dim=-1)
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
    

class llm_for_sgg(Base_LLM):
    def __init__(self, config, in_channels):
        self.logger = logging.getLogger(__name__)
        super().__init__(self.logger)
        
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

        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        rel_classes[rel_classes.index("__background__")]="background"
        obj_classes[obj_classes.index("__background__")]="background"
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        
        add_token_nums=self.add_token(rel_classes,['[CATE]','<roi>','</roi>','<p>','</p>'],self.logger)
        self.init_tokenizer_weight(num_new_tokens=add_token_nums,logger=self.logger)
        
        self.cate_tokenid = self.tokenizer('[CATE]', add_special_tokens=False).input_ids[-1]
        
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
        
        # ************************** LLM train module *********************************
        # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
        for n, p in self.lm.named_parameters():
            if any(
                [
                    x in n
                    for x in ["lm_head","embed_tokens"]
                ]
            ):
                self.logger.info(f"Calculate gradient name: {n}, param.shape: {p.shape}")
                p.requires_grad = True

        if self.lm.config.tune_mm_mlp_adapter:
            self.lm.requires_grad_(False)
            for p in self.lm.get_model().mm_projector.parameters():
                p.requires_grad = True
        else:
            for p in self.lm.get_model().mm_projector.parameters():
                p.requires_grad = False
                
        self.lm.to(dtype=self.torch_dtype, device=self.device)
        
        # ************************** project hidden states to relation module *********************************
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.img_proj = nn.Sequential(
            nn.Linear(roi_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.lm.config.hidden_size)
        )
        
        self.head_gate=nn.Sequential(
            nn.Linear(2*self.lm.config.hidden_size,self.lm.config.hidden_size),
            nn.Sigmoid()
        )
        self.tail_gate=nn.Sequential(
            nn.Linear(2*self.lm.config.hidden_size,self.lm.config.hidden_size),
            nn.Sigmoid()
        )
        
        self.head_linear_fuse=nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.lm.config.hidden_size,self.lm.config.hidden_size),
                nn.ReLU(inplace=True),
            ),
            nn.LayerNorm(self.lm.config.hidden_size)
        ])
        self.tail_linear_fuse=nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.lm.config.hidden_size,self.lm.config.hidden_size),
                nn.ReLU(inplace=True),
            ),
            nn.LayerNorm(self.lm.config.hidden_size)
        ])
        self.union_linear_fuse=nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.lm.config.hidden_size,self.lm.config.hidden_size),
                nn.ReLU(inplace=True),
            ),
            nn.LayerNorm(self.lm.config.hidden_size)
        ])
        
        self.rel_gate=nn.Sequential(
            nn.Linear(2*self.lm.config.hidden_size,self.lm.config.hidden_size),
            nn.Sigmoid()
        )
        self.rel_linear_fuse=nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.lm.config.hidden_size,self.lm.config.hidden_size),
                nn.ReLU(inplace=True),
            ),
            nn.LayerNorm(self.lm.config.hidden_size)
        ])
        
        self.mask_to_rel=MLP(self.lm.config.hidden_size,self.hidden_dim,self.num_rel_cls,1)
        self.proj_pred=MLP(self.lm.config.hidden_size,self.hidden_dim,self.lm.config.hidden_size, 2)

        self.init_weight(['img_proj','head_gate','tail_gate','head_linear_fuse','tail_linear_fuse','union_linear_fuse','rel_gate','rel_linear_fuse','mask_to_rel','proj_pred'])
        
    def init_weight(self,init_layers=[]):
        for name,param in self.named_parameters():
            if name.split('.')[0] in init_layers:
                layer_init(param,xavier=True)
                param.requires_grad=True
                param.data = param.data.to(self.device, dtype=self.torch_dtype)
                print(f'init weight for module: {name}, param shape: {param.shape}, detype: {param.dtype} require grad: {param.requires_grad}')
            
    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        # if not self.init_:
        #     self.init_weight(['img_proj','head_gate','tail_gate','head_linear_fuse','tail_linear_fuse','union_linear_fuse','rel_gate','rel_linear_fuse','mask_to_rel','proj_pred'])
    
        add_losses=dict()
        
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
        
        # ************************************************ LLM process relationship **********************************************************************
        rel_tokenizer=self.tokenizer(self.rel_classes, padding=True, return_tensors='pt').to(self.device)
        outputs = self.lm.model.model(
            input_ids=rel_tokenizer['input_ids'],
            attention_mask=rel_tokenizer['attention_mask'],
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=self.lm.config.output_attentions,
            output_hidden_states=self.lm.config.output_hidden_states,
            return_dict=True
        )
        encode_rel_states = outputs.last_hidden_state
        assert encode_rel_states.shape[1]==2, ValueError(f'LLM processed relation word features shape: {encode_rel_states.shape}')
        encode_rel_states=encode_rel_states[:,-1,:]
        
        # ************************************************************************************************************************************************************
        rel_dists,train_rel_labels = [],[]
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
            
            img_file=proposal.get_field("file_name")
            image=PIL.Image.open(img_file).convert("RGB")
            image=self.vision_processor.preprocess(image,return_tensors='pt')['pixel_values']
            image=image.to(self.device,dtype=self.torch_dtype)
            
            conv = conv_templates['llava_llama_2']
            conv.system="You are a helpful language and vision assistant. You can answer users' questions based on pictures and visual features of a location."
            header=f"I will give you a picture where the data in <roi></roi> is the visual feature in a certain area of the picture, and <p></p> is the question. Please answer the questions according to the picture: {DEFAULT_IMAGE_TOKEN}, and combined with the visual characteristics given in each question." 
    
            batch_roi_feature,batch_union_feature=self.img_proj(batch_roi_feature.to(self.device,dtype=self.torch_dtype)),self.img_proj(batch_union_feature.to(self.device,dtype=self.torch_dtype))
            
            if self.training:
                train_rel_nums,batch_rel_labels=20,rel_labels[batch_idx]
                fg_mask=rel_labels[batch_idx]>0
                fg_idx,bg_idx = torch.where(fg_mask)[0],torch.where(~fg_mask)[0]
                if fg_mask.sum()>train_rel_nums:
                    selected_fg_indices = fg_idx[torch.randperm(len(fg_idx))[:train_rel_nums]]
                    batch_rel_labels=batch_rel_labels[selected_fg_indices]
                    batch_rel_pair_idx=batch_rel_pair_idx[selected_fg_indices]
                    batch_union_feature=batch_union_feature[selected_fg_indices]
                else:
                    selected_bg_indices = bg_idx[torch.randperm(len(bg_idx))[:(train_rel_nums-fg_mask.sum())]]
                    batch_rel_labels=torch.cat([batch_rel_labels[fg_idx],batch_rel_labels[selected_bg_indices]],dim=0)
                train_rel_labels.append(batch_rel_labels)

                step_out_dict=self.forward_step(encode_rel_states,image,conv,header,batch_rel_pair_idx[:train_rel_nums],batch_roi_feature,batch_union_feature[:train_rel_nums,...],batch_rel_labels)
                rel_dists.append(step_out_dict.pop('rel_dists'))
                for loss_k,loss_v in step_out_dict['add_loss'].items():
                    add_losses[loss_k]=add_losses.get(loss_k,0.0)+loss_v
            else:
                split_num,split_rel_dists=30,[]
                for pair_id in range(math.ceil(batch_rel_pair_idx.shape[0]/split_num)):
                    step_out_dict=self.forward_step(encode_rel_states,image,conv,header,batch_rel_pair_idx[pair_id*split_num:(pair_id+1)*split_num],batch_roi_feature,batch_union_feature[pair_id*split_num:(pair_id+1)*split_num,...])
                    split_rel_dists.append(step_out_dict.pop('rel_dists'))

                rel_dists.append(torch.cat(split_rel_dists,dim=0).float())
        
        return entity_dists, rel_dists, add_losses, dict(train_rel_labels=train_rel_labels) if self.training else dict()
        
    def forward_step(self,encode_rel_states,image,conv,header,rel_pair_idx,roi_features,union_features,rel_labels=None):
        add_loss=dict()
        
        head_idx, tail_idx = rel_pair_idx[:, 0], rel_pair_idx[:, 1]
        head_obj_feature, tail_obj_feature = roi_features[head_idx], roi_features[tail_idx]
        
        question_templates,pl_answers,roi_features,gt_answers="","",[],[]
        for idx, (head_feature, tail_feature,union_feature) in enumerate(zip(head_obj_feature, tail_obj_feature,union_features)):
            question_templates += f" <p>In this <roi>{UNION_IMAGE_TOKEN}</roi>, what is the relationship between <roi>{UNION_IMAGE_TOKEN}</roi> and <roi>{UNION_IMAGE_TOKEN}</roi>?</p>"
            if self.training:
                gt_answers.append(self.rel_classes[rel_labels[idx]])
            pl_answers+='[CATE], '
            roi_features.append(torch.stack([union_feature,head_feature,tail_feature],dim=0))
            
        pl_answers=pl_answers[:-2]+"."
        
        conv.messages=[]
        conv.append_message(conv.roles[0],header+question_templates)
        conv.append_message(conv.roles[1],pl_answers)
        conv_prompt=conv.get_prompt()

        if self.training:
            input_ids,attention_masks,targets=self.process_target_conv(conv,conv_prompt,self.tokenizer)
            input_ids,attention_masks,targets=input_ids.to(device=self.device),attention_masks.to(device=self.device),targets.to(device=self.device)
            cate_row_index,cate_col_index=torch.where(input_ids==self.cate_tokenid)
            
            gt_cate_input_ids=self.tokenizer(gt_answers).input_ids
            gt_cate_input_ids=[item[-1] for item in gt_cate_input_ids]
            targets[cate_row_index,cate_col_index]=torch.tensor(gt_cate_input_ids,dtype=torch.long,device=self.device)
        else:
            input_ids=self.tokenizer_image_token(conv_prompt,self.tokenizer,return_tensors='pt').unsqueeze(0).to(device=self.device)
            cate_row_index,cate_col_index=torch.where(input_ids==self.cate_tokenid)

        generate_out=self.lm(input_ids,attention_mask=attention_masks if self.training else None,labels=targets if self.training else None,images=image,roi_features=torch.cat(roi_features,dim=0),output_hidden_states=True,return_dict=True)
        rel_mask_feature=generate_out.hidden_states[-1][cate_row_index,cate_col_index]
        
        mask_to_rel=self.mask_to_rel(rel_mask_feature)
        
        rel_mask_feature_norm=rel_mask_feature/rel_mask_feature.norm(dim=1,keepdim=True)
        encode_rel_cls_norm=encode_rel_states/encode_rel_states.norm(dim=1,keepdim=True)
        
        mask_rel_sim=rel_mask_feature_norm@encode_rel_cls_norm.t().contiguous()*self.logit_scale.exp()
        
        if self.training:
            add_loss['mask_to_rel']=add_loss.get('mask_to_rel',0.0)+F.cross_entropy(mask_to_rel,rel_labels)
            add_loss['mask_rel_sim']=add_loss.get('mask_rel_sim',0.0)+F.cross_entropy(mask_rel_sim,rel_labels)
        
        return dict(add_loss=add_loss,rel_dists=mask_rel_sim+mask_to_rel)
    
    def process_target_conv(self,conv,conversations,tokenizer):
        if not isinstance(conversations,(list,tuple)):
            conversations=[conversations]
        
        input_ids = [self.tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_masks = input_ids.ne(tokenizer.pad_token_id)
        targets = input_ids.clone()
        
        targets=self.process_target(conv,conversations,targets,tokenizer)
    
        return input_ids,attention_masks,targets

    def tokenizer_image_token(self,prompt, tokenizer, return_tensors=None):
        prompt_chunks ,input_ids= [],[]
        offset = 0

        match_img=re.split(DEFAULT_IMAGE_TOKEN,prompt)

        for idx,match_ in enumerate(match_img):
            if UNION_IMAGE_TOKEN in match_:
                match_bbox=re.split(UNION_IMAGE_TOKEN,match_)
                
                for b_idx,match_b in enumerate(match_bbox):
                    prompt_chunks.append(tokenizer(match_b).input_ids)
                    if b_idx!=len(match_bbox)-1:
                        prompt_chunks.append([UNION_IMAGE_INDEX])
            else:
                prompt_chunks.append(tokenizer(match_).input_ids)
                
            if idx!=len(match_img)-1:
                prompt_chunks.append([IMAGE_TOKEN_INDEX])

        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])
        
        for x in prompt_chunks:
            input_ids.extend(x[offset:] if len(x)>1 else x)

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    def process_target(self,conv,conversations,targets,tokenizer):
        # pdb.set_trace()
        sep = "[/INST] "
        for conversation, target in zip(conversations, targets):
            rounds = conversation.split(conv.sep2)  # 每段话分成问答对
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)  # 每对问答分成问和答

                assert len(parts) == 2, (len(parts), rou)
                parts[0] += sep

                if DEFAULT_IMAGE_TOKEN in conversation:
                    round_len =len(self.tokenizer_image_token(rou, tokenizer, return_tensors='pt'))
                    instruction_len =len(self.tokenizer_image_token(parts[0], tokenizer, return_tensors='pt')) - 2
            
                else:
                    round_len =len(self.tokenizer_image_token(rou, tokenizer, return_tensors='pt'))
                    instruction_len =len(self.tokenizer_image_token(parts[0], tokenizer, return_tensors='pt')) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
                
                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX
            
            if cur_len < tokenizer.model_max_length:
                # assert cur_len == total_len
                assert cur_len == len(target)
        
        return targets

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
    