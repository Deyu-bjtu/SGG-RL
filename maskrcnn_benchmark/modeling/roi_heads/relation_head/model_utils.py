import copy
import glob
import math
import os
import re
import time
import PIL
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from maskrcnn_benchmark.modeling.roi_heads.relation_head.llava_llama import LlavaLlamaForCausalLM
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_transformer import MultiHeadAttention, PositionwiseFeedForward
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_relation import layer_init
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.utils.comm import all_gather_with_grad, concat_all_gather, get_rank,find_linear_layers
from .utils_motifs import rel_vectors, obj_edge_vectors, to_onehot, nms_overlaps, encode_box_info 
from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.modeling.roi_heads.relation_head.conversation import conv_templates
from maskrcnn_benchmark.modeling.roi_heads.relation_head.llava_arch import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX, OBJECT_IMAGE_INDEX, SUBJECT_IMAGE_INDEX, UNION_IMAGE_INDEX,UNION_IMAGE_TOKEN,SUBJECT_IMAGE_TOKEN,OBJECT_IMAGE_TOKEN
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
            add_token_nums+= self.tokenizer.add_tokens(external_token)
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

        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
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
                
                extra_loss=self.calculate_semantic_loss(encode_rel_cls,encode_rel_cls_norm)
                extra_loss.update(self.calculate_similar_loss(encode_rel_cls,rel_mask_feature,batch_rel_labels))
                extra_loss.update(self.calculate_similar_loss(encode_rel_cls,visual_to_lg_rel_rep,batch_rel_labels,loss_name="visual_rel_dis"))
                
                for key,value in extra_loss.items():
                    add_losses[key]=add_losses.get(key,0.0)+value
                
            rel_dists.append(rel_rep_cls+mask_rel_sim)
            
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
        
        super(Base_LLM, self).__init__(self.logger)
        
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
        
        add_token_nums=self.add_token(rel_classes,['[CATE]'],self.logger)
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
        self.rel_hidden_fcs=nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.hidden_size, self.num_rels,bias=False)
        )
        layer_init(self.rel_hidden_fcs,xavier=True)
        self.rel_hidden_fcs.requires_grad_(True)
        self.rel_hidden_fcs.to(device=self.device,dtype=self.torch_dtype)
        
        
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
            
            image_features=self.lm.encode_images(image) 
            
            conv = conv_templates['llava_llama_2']
            split_question,split_gt_answer,split_pt_answer,split_union_feature,split_sub_feature,split_obj_feature=[],[],[],[],[],[]
            
            split_num=100
            for pair_id in range(math.ceil(batch_rel_pair_idx.shape[0]/split_num)):
                head_idx, tail_idx = batch_rel_pair_idx[pair_id*split_num:(pair_id+1)*split_num, 0], batch_rel_pair_idx[pair_id*split_num:(pair_id+1)*split_num, 1]
                
                if self.training:
                    gt_answers,pl_answers=[],""
                    for rel_l in rel_labels[batch_idx][pair_id*split_num:(pair_id+1)*split_num]:
                        gt_answers.append(rel_l)
                        pl_answers+='[CATE], '
                    gt_answers=gt_answers[:-2]+"."
                    pl_answers=pl_answers[:-2]+"."
                else:
                    pl_answers=""
                    for _ in range(head_idx.shape[0]):
                        pl_answers+='[CATE], '
                    pl_answers=pl_answers[:-2]+"."
                    
                head_obj_pre, tail_obj_pre = batch_obj_preds[head_idx], batch_obj_preds[tail_idx]
                head_obj_feature, tail_obj_feature = batch_roi_feature[head_idx], batch_roi_feature[tail_idx]

                question_templates=DEFAULT_IMAGE_TOKEN+ "\nBased on the above images, answer the following questions:" 

                for idx, (head_obj, tail_obj) in enumerate(zip(head_obj_pre, tail_obj_pre)):
                    question_templates += f"In {UNION_IMAGE_TOKEN}, {SUBJECT_IMAGE_TOKEN} is {self.obj_classes[head_obj]} and {OBJECT_IMAGE_TOKEN} is {self.obj_classes[tail_obj]}. What is the relationship between {self.obj_classes[head_obj]} and {self.obj_classes[tail_obj]}?"
                        
                # split_question.append(question_templates)
                # split_union_feature.append(batch_union_feature[pair_id*split_num:(pair_id+1)*split_num,...])
                # split_sub_feature.append(head_obj_feature)
                # split_obj_feature.append(tail_obj_feature)

                roi_features=dict(DEFAULT_IMAGE_TOKEN=[image_features],UNION_IMAGE_TOKEN=batch_union_feature[pair_id*split_num:(pair_id+1)*split_num,...],SUBJECT_IMAGE_TOKEN=head_obj_feature,OBJECT_IMAGE_TOKEN=tail_obj_feature)
                
                conv.messages=[]
                conv.append_message(conv.roles[0],question_templates)
                conv.append_message(conv.roles[1],pl_answers)
                conv_prompt=conv.get_prompt()
                
                input_ids,attention_masks,targets,cur_new_input_embeds=self.process_target_conv(conv,conv_prompt,self.tokenizer,roi_features)
                cate_row_index,cate_col_index=torch.where(input_ids==self.cate_tokenid)
                
                if self.training:
                    gt_cate_input_ids=self.tokenizer(gt_answers).input_ids
                    gt_cate_input_ids=[item[-1] for item in gt_cate_input_ids]
                    targets[cate_row_index,cate_col_index]=torch.tensor(gt_cate_input_ids,dtype=torch.long)
                
                input_ids,attention_masks,targets=input_ids.to(device=self.device),attention_masks.to(device=self.device),targets.to(device=self.device)

                generate_out=self.lm(input_ids,attention_mask=attention_masks,labels=targets,images=image,output_hidden_states=True,return_dict=True)
     
      
    def process_target_conv(self,conv,conversations,tokenizer,features=None):
        if not isinstance(conversations,(list,tuple)):
            conversations=[conversations]

        if features is not None:
            assert len(conversations)==1
        
        input_ids,cur_new_input_embeds,cur_input_positions=[],[],[]
        for prompt in conversations:
            input_id,cur_new_input_embed,cur_input_pos=self.tokenizer_image_token(prompt, tokenizer, features, return_tensors='pt')
            input_ids.append(input_id)
            cur_new_input_embeds.append(cur_new_input_embed)
            cur_input_positions.append(cur_input_pos)
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_masks = input_ids.ne(tokenizer.pad_token_id)
        targets = input_ids.clone()
        cur_new_input_embeds=torch.stack(cur_new_input_embeds,dim=0)
        
        assert cur_new_input_embeds.shape[0]==input_ids.shape[0]
        
        new_attn_mask_pad_left = torch.full((attention_masks.shape[0], cur_new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_masks.dtype, device=attention_masks.device)
        attention_masks = torch.cat((new_attn_mask_pad_left, attention_masks), dim=1)
        assert attention_masks.shape == cur_new_input_embeds.shape[:2]

        targets=self.process_target_llama_2(conv,conversations,targets,tokenizer)
        for target,cur_input_pos in zip(targets,cur_input_positions):
            pass        
    
        return input_ids,attention_masks,targets,cur_new_input_embeds

    def tokenizer_image_token(self,prompt, tokenizer, features, return_tensors=None):
        prompt_chunks ,input_ids, cur_new_input_embeds,cur_input_pos= [],[],[],[]
        offset = 0
        
        special_tokens = {
            DEFAULT_IMAGE_TOKEN: IMAGE_TOKEN_INDEX,
            UNION_IMAGE_TOKEN: UNION_IMAGE_INDEX,
            SUBJECT_IMAGE_TOKEN: SUBJECT_IMAGE_INDEX,
            OBJECT_IMAGE_TOKEN: OBJECT_IMAGE_INDEX
        }

        start=0
        for match in re.finditer(r"<(image|union|sub|obj)>", prompt):
            prompt_chunks.append(tokenizer(prompt[start:match.start()], add_special_tokens=False).input_ids)
            prompt_chunks.append([special_tokens[match.group()]])
            
            if prompt_chunks[-1][0] == tokenizer.bos_token_id:
                offset=1
                    
            if len(cur_input_pos)==0:
                cur_input_pos.append(len(tokenizer(prompt[start:match.start()], add_special_tokens=False).input_ids))
            else:
                cur_input_pos.append(len(tokenizer(prompt[start:match.start()], add_special_tokens=False).input_ids)-offset)
            
            if features is not None:
                cur_new_input_embeds.append(self.lm.get_model().embed_tokens(torch.tensor(prompt_chunks[-1][offset:],dtype=torch.long)).to(device=self.device))
                cur_new_input_embeds.append(features[match.group()].pop(0).to(device=self.device))
                
            start = match.end()
        
        prompt_chunks.append(tokenizer(prompt[start:], add_special_tokens=False).input_ids)
        cur_input_pos.append(len(tokenizer(prompt[start:], add_special_tokens=False).input_ids)-offset)
        if features is not None:
            cur_new_input_embeds.append(self.lm.get_model().embed_tokens(torch.tensor(prompt_chunks[-1][offset:],dtype=torch.long)).to(device=self.device))
        if offset==1:
            cur_new_input_embeds.insert(0,self.lm.get_model().embed_tokens(torch.tensor(tokenizer.bos_token_id,dtype=torch.long)).to(device=self.device))
            
        cur_new_input_embeds=torch.cat(cur_new_input_embeds,dim=0)
        
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])
        
        for x in prompt_chunks:
            input_ids.extend(x[offset:] if len(x)>1 else x)

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        
        special_token_nums=sum(prompt==DEFAULT_IMAGE_TOKEN)+sum(prompt==UNION_IMAGE_TOKEN)+sum(prompt==SUBJECT_IMAGE_TOKEN)+sum(prompt==OBJECT_IMAGE_TOKEN)
        assert sum(cur_input_pos)+special_token_nums==len(input_ids)
        return input_ids,cur_new_input_embeds,cur_input_pos

    def process_target_llama_2(self,conv,conversations,targets,tokenizer):
        sep = "[/INST] "
        for conversation, target in zip(conversations, targets):
            # total_len = int(target.ne(tokenizer.pad_token_id).sum())
            
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
                    round_len =len(self.tokenizer_image_token()(rou, tokenizer, return_tensors='pt'))
                    instruction_len =len(self.tokenizer_image_token(parts[0], tokenizer, return_tensors='pt')) - 2
            
                else:
                    raise

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
    def __init__(self, max_features_per_class, in_feature_dim,rel_cls_num,save_dir,out_feature_dim=256):
        super().__init__()
        self.max_features_per_class = max_features_per_class
        self.sample_feature=nn.Linear(in_feature_dim,out_feature_dim)
        self.memory_bank = {cls_id:[] for cls_id in range(rel_cls_num)}
        self.save_dir=save_dir
        self.check_device=False
        
        if os.path.exists(f'{self.save_dir}/memory_features.pth'):
            self.memory_bank=torch.load(f'{self.save_dir}/memory_features.pth',map_location='cpu')
            self.check_device=True

    def add_feature(self, class_ids, features):
        if self.check_device:
            device=features.device
            for name,value in self.memory_bank.items():
                for idx,feature in enumerate(value):
                    value[idx]=feature.to(device)
                self.memory_bank[name]=value
            self.check_device=False
        
        detach_features=features.clone().detach()
        detach_features=self.sample_feature(detach_features)
        
        for class_id,feature in zip(class_ids,detach_features):
            class_id=class_id.item()
    
            if len(self.memory_bank[class_id]) < self.max_features_per_class:
                self.memory_bank[class_id].append(feature)
            else:
                self.memory_bank[class_id].pop(0)
                self.memory_bank[class_id].append(feature)

        # print({cls_id: len(values) for cls_id,values in self.memory_bank.items()})
        
    def optim_sample_feature(self,rel_rep,rel_labels,margin=0.1):
        for memory_features in self.memory_bank.values():
            if len(memory_features)==0:
                return None
        torch.save(self.memory_bank,f'{self.save_dir}/memory_features_{torch.cuda.current_device()}.pth')
        
        all_memory_features=[torch.stack(value,dim=0).mean(dim=0) for value in self.memory_bank.values()]
        all_memory_features=torch.stack(all_memory_features,dim=0)
        all_memory_features_norm=all_memory_features/all_memory_features.norm(dim=-1,keepdim=True)
        
        detach_rel_rep=rel_rep.clone().detach()
        detach_rel_rep=self.sample_feature(detach_rel_rep)
                
        add_loss=self.calculate_semantic_loss(all_memory_features,all_memory_features_norm)
        add_loss.update(self.calculate_similar_loss(all_memory_features,detach_rel_rep,rel_labels))
        
        rel_rep_sim_matrix=F.normalize(detach_rel_rep,dim=-1)@all_memory_features_norm.t().contiguous()
        
        pos_samples = rel_rep_sim_matrix[torch.arange(rel_rep_sim_matrix.shape[0]), rel_labels].view(rel_rep_sim_matrix.shape[0],-1)
        
        neg_matrix=torch.ones_like(rel_rep_sim_matrix,dtype=torch.long)
        neg_matrix[torch.arange(rel_rep_sim_matrix.shape[0]), rel_labels] = 0
        neg_samples=rel_rep_sim_matrix[neg_matrix].view(rel_rep_sim_matrix.shape[0],-1)
        
        add_loss['sim_loss']=add_loss.get('sim_loss',0.0)+torch.mean((1 - pos_samples) ** 2)+torch.mean(F.relu(neg_samples - margin) ** 2)
        
        return add_loss
    
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
    
    def predict_similarity(self, rel_rep):
        for memoey_features in self.memory_bank.values():
            if len(memoey_features)==0:
                return None
        
        all_memory_features=[torch.stack(value,dim=0).mean(dim=0) for value in self.memory_bank.values()]
        all_memory_features=torch.stack(all_memory_features,dim=0)
        all_memory_features_norm=all_memory_features/all_memory_features.norm(dim=-1,keepdim=True)
        
        rel_rep=self.sample_feature(rel_rep)
        rel_rep_sim_matrix=F.normalize(rel_rep,dim=-1)@all_memory_features_norm.t().contiguous()
        
        return rel_rep_sim_matrix
