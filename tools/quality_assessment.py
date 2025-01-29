# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
import functools
import os,sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.abspath(os.path.join(current_dir,'../')))
from glob import glob
import argparse
import time
import torch
from tqdm import tqdm
from tools.relation_train_net import fix_eval_modules
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.comm import synchronize
import numpy
import random
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import CCA
from openpyxl import Workbook

head=['on','has','wearing','of','in','near','behind']
body=['with','holding','above','under','wears','sitting on','in front of','riding','standing on','at',
        'attached to','over','carrying','walking on','for','looking at','watching','hanging from','belonging to',
        'and','parked on']
tail=['between','laying on','along','eating','covering','covered in','part of','using','to','on back of',
        'across','mounted on','lying on','walking in','against','from','growing on','painted on','made of',
        'playing','says','flying in']
idx_to_predicate={"1": "above", "2": "across", "3": "against", "4": "along", "5": "and", "6": "at", "7": "attached to", "8": "behind", "9": "belonging to", "10": "between", "11": "carrying", "12": "covered in", "13": "covering", "14": "eating", "15": "flying in", "16": "for", "17": "from", "18": "growing on", "19": "hanging from", "20": "has", "21": "holding", "22": "in", "23": "in front of", "24": "laying on", "25": "looking at", "26": "lying on", "27": "made of", "28": "mounted on", "29": "near", "30": "of", "31": "on", "32": "on back of", "33": "over", "34": "painted on", "35": "parked on", "36": "part of", "37": "playing", "38": "riding", "39": "says", "40": "sitting on", "41": "standing on", "42": "to", "43": "under", "44": "using", "45": "walking in", "46": "walking on", "47": "watching", "48": "wearing", "49": "wears", "50": "with"}
    

def load_reps():
    sample_ids=os.listdir("/data/sdc/checkpoints/SGG_Benchmark/VG/PE_V2_predcls_detach_relcenter_withbias_withPCR_without_Lcs_Lpc/features/")
    dpplml_path,pe_path='/data/sdc/checkpoints/SGG_Benchmark/VG/PE_V2_predcls_detach_relcenter_withbias_withPCR_without_Lcs_Lpc/features/','/data/sdc/checkpoints/SGG_Benchmark/VG/PE_V2_predcls_detach_relcenter_withbias_withPCR_without_Lcs_Lpc/pe_features'
    
    pe_sub_emb,pe_obj_emb,pe_rel_reps,pe_rel_labels=[],[],[],[]
    
    dpplml_sub_emb,dpplml_obj_emb,dpplml_entity_rel_rep=[],[],[]
    dpplml_s_p_rep,dpplml_o_p_rep,dpplml_rel_rep,dpplml_rel_center,dpplml_rel_labels=[],[],[],[],[]
    # for sample_id in tqdm(random.sample(sample_ids,k=1000)):
    for sample_id in tqdm(sample_ids):
        # ************ *************** ************
        dpplml_entity_dicts=torch.load(f'{dpplml_path}/{sample_id}/entity_reps.pth',map_location='cpu')
        dpplml_rel_dicts=torch.load(f'{dpplml_path}/{sample_id}/rel_reps.pth',map_location='cpu')
        
        dpplml_labels=dpplml_rel_dicts['rel_label'].cpu()
        
        # ************ PE Net Features ************
        pe_entity_dicts=torch.load(f'{pe_path}/{sample_id}/entity_features.pth',map_location='cpu')
        pe_rel_dicts=torch.load(f'{pe_path}/{sample_id}/rel_features.pth',map_location='cpu')
        
        pe_labels=pe_rel_dicts['rel_labels'].cpu()
        
        # ************ *************** ************
        # continue_epoch=False
        assert len(pe_labels)==len(dpplml_labels)
        # for pe_l,dpplml_l in zip(pe_rel_dicts['rel_labels'].cpu(),dpplml_labels[fg_mask]):
        #     if not torch.equal(pe_l,dpplml_l):
        #         print(f'PE Net labels: {pe_rel_dicts["rel_labels"]}, DPPLML labels: {dpplml_labels[fg_mask]}')
        #         continue_epoch=True
        #         break
        
        # if continue_epoch:
        #     continue
        
        # ************ insert data ************
        
        dpplml_sub_emb.append(dpplml_entity_dicts['sub_embeds'].cpu())
        dpplml_obj_emb.append(dpplml_entity_dicts['obj_embeds'].cpu())
        dpplml_entity_rel_rep.append(dpplml_entity_dicts['rel_reps'])
        
        dpplml_s_p_rep.append(dpplml_rel_dicts['sp_query'].cpu())
        dpplml_o_p_rep.append(dpplml_rel_dicts['op_query'].cpu())
        dpplml_rel_rep.append(dpplml_rel_dicts['rel_query'].cpu())
        dpplml_rel_center.append(dpplml_rel_dicts['rel_center'].cpu())  # all_samples,num_rels,hidden_dim
        dpplml_pro=dpplml_rel_dicts['predicate_prototye'].cpu() # num_rels,hidden_dim
        dpplml_rel_labels.append(dpplml_labels)
        
        pe_sub_emb.append(pe_entity_dicts['sub_sem_reps'].cpu())
        pe_obj_emb.append(pe_entity_dicts['obj_sem_reps'].cpu())
        pe_pro=pe_rel_dicts['predicate_proto'].cpu()
        
        pe_rel_reps.append(pe_rel_dicts['rel_sem_reps'].cpu())
        pe_rel_labels.append(pe_labels)
        
        null_cls=False
        for rel_id in range(1,51):
            exist_labels=torch.cat(dpplml_rel_labels,dim=0)
            if torch.sum(exist_labels==rel_id)<30:
                null_cls=True
                break
        
        if not null_cls:
            print(f'Each class have features more than fifty.')
            break
             
    
    return torch.cat(dpplml_sub_emb,dim=0),torch.cat(dpplml_obj_emb,dim=0),torch.cat(dpplml_entity_rel_rep,dim=0),torch.cat(dpplml_s_p_rep,dim=0),torch.cat(dpplml_o_p_rep,dim=0),torch.cat(dpplml_rel_rep,dim=0),torch.cat(dpplml_rel_center,dim=0),dpplml_pro,torch.cat(dpplml_rel_labels,dim=0),torch.cat(pe_sub_emb,dim=0),torch.cat(pe_obj_emb,dim=0),torch.cat(pe_rel_reps,dim=0),pe_pro,torch.cat(pe_rel_labels,dim=0)

def load_reps_for_multistep(base_name,base_path='/opt/data/private/zgq/SGG_Benchmark/reps_space'):
    sample_ids=os.listdir(base_path)
    
    ori,step1,step2=dict(),dict(),dict()
    ori_save_nums,step1_save_nums,step2_save_nums=dict(),dict(),dict()
    for i in range(1,51):
        ori_save_nums[i]=0
        step1_save_nums[i]=0
        step2_save_nums[i]=0
    # for sample_id in tqdm(random.sample(sample_ids,k=10000)):
    for sample_id in tqdm(sample_ids):
        if not os.path.isdir(f'{base_path}/{sample_id}'):
            continue
        # ************ *************** ************
        if os.path.exists(f'{base_path}/{sample_id}/{base_name}_None_step_1.pt'):
            none_infos=torch.load(f'{base_path}/{sample_id}/{base_name}_None_step_1.pt',map_location='cpu')
        else:
            raise ValueError(f'{base_path}/{sample_id}/{base_name}_None_step_1.pt is not found!!')
        
        if "PENet" in base_name:
            if os.path.exists(f'{base_path}/{sample_id}/{base_name}_Multi_step_Denoise_step_1.pt'):
                step1_infos=torch.load(f'{base_path}/{sample_id}/{base_name}_v2_Multi_step_Denoise_step_1.pt',map_location='cpu')
            else:
                raise ValueError(f'{base_path}/{sample_id}/{base_name}_v2_Multi_step_Denoise_step_1.pt is not found!!')
            
            if os.path.exists(f'{base_path}/{sample_id}/{base_name}_v2_Multi_step_Denoise_step_2.pt'):
                step2_infos=torch.load(f'{base_path}/{sample_id}/{base_name}_v2_Multi_step_Denoise_step_2.pt',map_location='cpu')
            else:
                step2_infos=None
        else:
            if os.path.exists(f'{base_path}/{sample_id}/{base_name}_Multi_step_Denoise_step_1.pt'):
                step1_infos=torch.load(f'{base_path}/{sample_id}/{base_name}_Multi_step_Denoise_step_1.pt',map_location='cpu')
            else:
                raise ValueError(f'{base_path}/{sample_id}/{base_name}_Multi_step_Denoise_step_1.pt is not found!!')
            
        #     if os.path.exists(f'{base_path}/{sample_id}/{base_name}_Multi_step_Denoise_step_2.pt'):
        #         step2_infos=torch.load(f'{base_path}/{sample_id}/{base_name}_Multi_step_Denoise_step_2.pt',map_location='cpu')
        #     else:
        #         step2_infos=None
                
        # if step2_infos is None:
        #     print(f'{sample_id}_Multi_step_Denoise_step_2 is not exits')
        #     continue
        
        for rel_id in range(1,51):
            if ori_save_nums[rel_id]<10000:
                save_info,save_len=load_transformer_reps(rel_id,none_infos,ori,'None')
                if save_info is not None:
                    ori=save_info
                    ori_save_nums[rel_id]=ori_save_nums[rel_id]+save_len
                
            min_idx=min(list(ori_save_nums.values()))//1000
            if min_idx>=1:
                if not os.path.exists(f'{base_path}/{base_name}_None_{min_idx}.pt'):
                    torch.save(dict(infos=ori,nums=ori_save_nums),f'{base_path}/{base_name}_None_{min_idx}.pt')
                    print(f'save {base_name}_None_{min_idx} predict infos success, min len: {min(list(ori_save_nums.values()))}.....')
            
            if step1_save_nums[rel_id]<10000:
                save_info,save_len=load_transformer_reps(rel_id,step1_infos,step1,'step1')
                if save_info is not None:
                    step1=save_info
                    step1_save_nums[rel_id]=step1_save_nums[rel_id]+save_len
                
            min_idx=min(list(step1_save_nums.values()))//1000
            if min_idx>=1:
                if not os.path.exists(f'{base_path}/{base_name}_step1_{min_idx}.pt'):
                    torch.save(dict(infos=step1,nums=step1_save_nums),f'{base_path}/{base_name}_step1_{min_idx}.pt')
                    print(f'save {base_name}_step1_{min_idx} predict infos success, min len: {min(list(step1_save_nums.values()))}.....')
                    
            # if step2_save_nums[rel_id]<10000:
            #     try:
            #         save_info,save_len=load_transformer_reps(rel_id,step2_infos,step2,'step2')
            #     except:
            #         save_info,save_len=None,0
            #     if save_info is not None:
            #         step2=save_info
            #         step2_save_nums[rel_id]=step2_save_nums[rel_id]+save_len
                    
            # min_idx=min(list(step2_save_nums.values()))//1000
            # if min_idx>=1:
            #     if not os.path.exists(f'{base_path}/{base_name}_step2_{min_idx}.pt'):
            #         torch.save(dict(infos=step2,nums=step2_save_nums),f'{base_path}/{base_name}_step2_{min_idx}.pt')
            #         print(f'save {base_name}_step2_{min_idx} predict infos success, min len: {min(list(step2_save_nums.values()))}.....')
            
    torch.save(dict(infos=ori,nums=ori_save_nums),f'{base_path}/{base_name}_None.pt')
    print(f'save {base_name}_None predict infos success.....')
    torch.save(dict(infos=step1,nums=step1_save_nums),f'{base_path}/{base_name}_step1.pt')
    print(f'save {base_name}_step1 predict infos success.....')
    # torch.save(dict(infos=step2,nums=step2_save_nums),f'{base_path}/{base_name}_step2.pt')
    # print(f'save {base_name}_step2 predict infos success.....')
    
             
def load_transformer_reps(rel_id,load_infos,save_infos,types='None'):
    sub_,obj_,prod_,vis_,label=load_infos['overall']['sub_embed'].cpu(),load_infos['overall']['obj_embeds'].cpu(),load_infos['overall']['prod_rep'].cpu(),load_infos['overall']['vis_rep'].cpu(),load_infos['overall']['rel_label'].cpu()
    # if len(torch.where(label==rel_id))<5:
    #     return None,0
    sample_idx=torch.where(label==rel_id)
    sub_,obj_,prod_,vis_=sub_[sample_idx],obj_[sample_idx],prod_[sample_idx],vis_[sample_idx]
    if rel_id not in save_infos:
        this_rel_infos=dict(sub_embeds=sub_,obj_embeds=obj_,prod_rep=prod_,vis_rep=vis_)
    else:
        this_rel_infos=save_infos[rel_id]
        this_rel_infos['sub_embeds']=torch.cat([this_rel_infos['sub_embeds'],sub_],dim=0)
        this_rel_infos['obj_embeds']=torch.cat([this_rel_infos['obj_embeds'],obj_],dim=0)
        this_rel_infos['prod_rep']=torch.cat([this_rel_infos['prod_rep'],prod_],dim=0)
        this_rel_infos['vis_rep']=torch.cat([this_rel_infos['vis_rep'],vis_],dim=0)
    
    if types=='step1':
        this_rel_infos=load_step1_infos(sample_idx,load_infos,this_rel_infos)
    if types=='step2':
        this_rel_infos=load_step1_infos(sample_idx,load_infos,this_rel_infos)
        this_rel_infos=load_step2_infos(sample_idx,load_infos,this_rel_infos)
    
    save_infos[rel_id]=this_rel_infos    
    return save_infos,len(torch.where(label==rel_id)) 

def load_penet_reps(rel_id,load_infos,save_infos,types='None'):
    sub_,obj_,rep_,proto_,label,fusion_=load_infos['overall']['sub_embed'].cpu(),load_infos['overall']['obj_embeds'].cpu(),load_infos['overall']['rel_rep'].cpu(),load_infos['overall']['predicate_proto'].cpu(),load_infos['overall']['rel_label'].cpu(),load_infos['overall']['fusion_so'].cpu()
    # if len(torch.where(label==rel_id))<5:
    #     return None,0
    sample_idx=torch.where(label==rel_id)
    sub_,obj_,rep_,proto_,fusion_=sub_[sample_idx],obj_[sample_idx],rep_[sample_idx],proto_[rel_id],fusion_[sample_idx]
    if rel_id not in save_infos:
        this_rel_infos=dict(sub_embeds=sub_,obj_embeds=obj_,rel_reps=rep_,fusion_so=fusion_,proto=proto_)
    else:
        this_rel_infos=save_infos[rel_id]
        this_rel_infos['sub_embeds']=torch.cat([this_rel_infos['sub_embeds'],sub_],dim=0)
        this_rel_infos['obj_embeds']=torch.cat([this_rel_infos['obj_embeds'],obj_],dim=0)
        this_rel_infos['rel_reps']=torch.cat([this_rel_infos['rel_reps'],rep_],dim=0)
        this_rel_infos['fusion_so']=torch.cat([this_rel_infos['fusion_so'],fusion_],dim=0)
    
    if types=='step1':
        this_rel_infos=load_step1_infos(sample_idx,load_infos,this_rel_infos)
    if types=='step2':
        this_rel_infos=load_step1_infos(sample_idx,load_infos,this_rel_infos)
        this_rel_infos=load_step2_infos(sample_idx,load_infos,this_rel_infos)
    
    save_infos[rel_id]=this_rel_infos    
    return save_infos,len(torch.where(label==rel_id))
    
def load_step1_infos(sample_idx,load_infos,this_rel_infos):
    sub_node,obj_node,refine_edg_rel_rep,filter_denoised_edg_rel_rep,proj_edg_rel_reps,noised_tri_rel_rep,denoise_tri_rel_rep,proj_denoise_tri_rel_rep,filter_denoised_tri_rel_rep,sum_rel_rep,glob_rel_rep=load_infos['step1']['sub_node'],load_infos['step1']['obj_node'],load_infos['step1']['refine_edg_rel_rep'],load_infos['step1']['filter_denoised_edg_rel_rep'],load_infos['step1']['proj_edg_rel_reps'],load_infos['step1']['noised_tri_rel_rep'],load_infos['step1']['denoise_tri_rel_rep'],load_infos['step1']['proj_denoise_tri_rel_rep'],load_infos['step1']['filter_denoised_tri_rel_rep'],load_infos['step1']['sum_rel_rep'],load_infos['step1']['glob_rel_rep']
    
    if 'sub_nodes' not in this_rel_infos.keys():
        this_rel_infos['sub_nodes']=sub_node[sample_idx]
    else:
        this_rel_infos['sub_nodes']=torch.cat([this_rel_infos['sub_nodes'],sub_node[sample_idx]],dim=0)
    if 'obj_nodes' not in this_rel_infos.keys():
        this_rel_infos['obj_nodes']=obj_node[sample_idx]
    else:
        this_rel_infos['obj_nodes']=torch.cat([this_rel_infos['obj_nodes'],obj_node[sample_idx]],dim=0) 
    if 'refine_edg_rel_reps' not in this_rel_infos.keys():
        this_rel_infos['refine_edg_rel_reps']=refine_edg_rel_rep[sample_idx]
    else:
        this_rel_infos['refine_edg_rel_reps']=torch.cat([this_rel_infos['refine_edg_rel_reps'],refine_edg_rel_rep[sample_idx]],dim=0)
    if 'filter_denoised_edg_rel_reps' not in this_rel_infos.keys():
        this_rel_infos['filter_denoised_edg_rel_reps']=filter_denoised_edg_rel_rep[sample_idx]
    else:
        this_rel_infos['filter_denoised_edg_rel_reps']=torch.cat([this_rel_infos['filter_denoised_edg_rel_reps'],filter_denoised_edg_rel_rep[sample_idx]],dim=0)
    if 'proj_edg_rel_reps' not in this_rel_infos.keys():
        this_rel_infos['proj_edg_rel_reps']=proj_edg_rel_reps[sample_idx]
    else:
        this_rel_infos['proj_edg_rel_reps']=torch.cat([this_rel_infos['proj_edg_rel_reps'],proj_edg_rel_reps[sample_idx]],dim=0)
    if 'noised_tri_rel_reps' not in this_rel_infos.keys():
        this_rel_infos['noised_tri_rel_reps']=noised_tri_rel_rep[sample_idx]
    else:
        this_rel_infos['noised_tri_rel_reps']=torch.cat([this_rel_infos['noised_tri_rel_reps'],noised_tri_rel_rep[sample_idx]],dim=0)
    if 'denoise_tri_rel_reps' not in this_rel_infos.keys():
        this_rel_infos['denoise_tri_rel_reps']=denoise_tri_rel_rep[sample_idx]
    else:
        this_rel_infos['denoise_tri_rel_reps']=torch.cat([this_rel_infos['denoise_tri_rel_reps'],denoise_tri_rel_rep[sample_idx]],dim=0)
    if 'proj_denoise_tri_rel_reps' not in this_rel_infos.keys():
        this_rel_infos['proj_denoise_tri_rel_reps']=proj_denoise_tri_rel_rep[sample_idx]
    else:
        this_rel_infos['proj_denoise_tri_rel_reps']=torch.cat([this_rel_infos['proj_denoise_tri_rel_reps'],proj_denoise_tri_rel_rep[sample_idx]],dim=0)
    if 'filter_denoised_tri_rel_reps' not in this_rel_infos.keys():
        this_rel_infos['filter_denoised_tri_rel_reps']=filter_denoised_tri_rel_rep[sample_idx]
    else:
        this_rel_infos['filter_denoised_tri_rel_reps']=torch.cat([this_rel_infos['filter_denoised_tri_rel_reps'],filter_denoised_tri_rel_rep[sample_idx]],dim=0)
    if 'sum_rel_reps' not in this_rel_infos.keys():
        this_rel_infos['sum_rel_reps']=sum_rel_rep[sample_idx]
    else:
        this_rel_infos['sum_rel_reps']=torch.cat([this_rel_infos['sum_rel_reps'],sum_rel_rep[sample_idx]],dim=0)
    if 'glob_rel_reps' not in this_rel_infos.keys():
        this_rel_infos['glob_rel_reps']=glob_rel_rep[sample_idx]
    else:
        this_rel_infos['glob_rel_reps']=torch.cat([this_rel_infos['glob_rel_reps'],glob_rel_rep[sample_idx]],dim=0)

    if 'step1_proto' not in this_rel_infos.keys() and 'rel_proto' in load_infos['step1'].keys():
        this_rel_infos['step1_proto']=load_infos['step1']['rel_proto']
    return this_rel_infos

def load_step2_infos(sample_idx,load_infos,this_rel_infos):
    encode_ctx,mean,std,x_T,recon_q_ctx,qu_proj_ctx,decode_ctx=load_infos['step2']['encode_ctx'],load_infos['step2']['mean'],load_infos['step2']['std'],load_infos['step2']['x_T'],load_infos['step2']['recon_q_ctx'],load_infos['step2']['qu_proj_ctx'],load_infos['step2']['decode_ctx']
    
    if 'encode_ctx' not in this_rel_infos.keys():
        this_rel_infos['encode_ctx']=encode_ctx[sample_idx]
    else:
        this_rel_infos['encode_ctx']=torch.cat([this_rel_infos['encode_ctx'],encode_ctx[sample_idx]],dim=0)
    if 'mean' not in this_rel_infos.keys():
        this_rel_infos['mean']=mean[sample_idx]
    else:
        this_rel_infos['mean']=torch.cat([this_rel_infos['mean'],mean[sample_idx]],dim=0) 
    if 'std' not in this_rel_infos.keys():
        this_rel_infos['std']=std[sample_idx]
    else:
        this_rel_infos['std']=torch.cat([this_rel_infos['std'],std[sample_idx]],dim=0)
    if 'x_T' not in this_rel_infos.keys():
        this_rel_infos['x_T']=x_T[sample_idx]
    else:
        this_rel_infos['x_T']=torch.cat([this_rel_infos['x_T'],x_T[sample_idx]],dim=0)
    if 'recon_q_ctx' not in this_rel_infos.keys():
        this_rel_infos['recon_q_ctx']=recon_q_ctx[sample_idx]
    else:
        this_rel_infos['recon_q_ctx']=torch.cat([this_rel_infos['recon_q_ctx'],recon_q_ctx[sample_idx]],dim=0)
    if 'qu_proj_ctx' not in this_rel_infos.keys():
        this_rel_infos['qu_proj_ctx']=qu_proj_ctx[sample_idx]
    else:
        this_rel_infos['qu_proj_ctx']=torch.cat([this_rel_infos['qu_proj_ctx'],qu_proj_ctx[sample_idx]],dim=0)
    if 'decode_ctx' not in this_rel_infos.keys():
        this_rel_infos['decode_ctx']=decode_ctx[sample_idx]
    else:
        this_rel_infos['decode_ctx']=torch.cat([this_rel_infos['decode_ctx'],decode_ctx[sample_idx]],dim=0)
    
    if 'step2_encode_proto' not in this_rel_infos.keys():
        this_rel_infos['step2_encode_proto']=load_infos['step2']['encode_proto']
    
    if 'step2_decode_proto' not in this_rel_infos.keys():
        this_rel_infos['step2_decode_proto']=load_infos['step2']['decode_proto']
    
    if 'q_embed' not in this_rel_infos.keys():
        this_rel_infos['q_embed']=load_infos['step2']['q_embed']
    
    return this_rel_infos
    
def vis_sim_matrix(x_reps,y_reps,x_name,y_name,file_name):
    if os.path.isfile(file_name):
        os.remove(file_name)
        
    from sklearn.metrics.pairwise import cosine_similarity
    
    plt.figure(figsize=(10, 10))

    vmin,vmax=-15,15

    similarity_matrix = cosine_similarity(x_reps, y_reps)

    dia_matrix=np.zeros(similarity_matrix.shape)
    np.fill_diagonal(dia_matrix,1)

    sns.heatmap(similarity_matrix, vmin=vmin, vmax=vmax, cmap='coolwarm', xticklabels=False, yticklabels=False)
    plt.xlabel(x_name, fontsize=18)
    plt.ylabel(y_name, fontsize=18)
    plt.savefig(file_name, bbox_inches='tight')
    plt.clf()

def vis_dis_matrix(x_reps,y_reps,x_name,y_name,file_name,add_info=None):
    if os.path.isfile(file_name):
        os.remove(file_name)
        
    # x_reps,y_reps=F.normalize(x_reps,p=2,dim=1),F.normalize(y_reps,p=2,dim=1)
    
    import seaborn as sns
    from scipy.spatial.distance import cdist

    distance_matrix = cdist(x_reps, y_reps, metric='cityblock')

    if add_info is not None:
        distance_matrix+=add_info.numpy()

    sns.heatmap(distance_matrix, cmap='coolwarm', xticklabels=False, yticklabels=False)
    plt.xlabel(x_name, fontsize=18)
    plt.ylabel(y_name, fontsize=18)
    plt.show()
    plt.savefig(file_name, bbox_inches='tight')
    plt.clf()

# 类内方差
def intra_inter_var(reps,rel_pros,labels,cls_num=51):
    wcv,bcv=[],[]
    overall_center = torch.mean(reps, dim=0)
    for i in range(1,cls_num):
        cls_points=reps[labels==i]
        cls_center = rel_pros[i]
        wcv.append(torch.sum((cls_points - cls_center) ** 2).item()/len(cls_points))
        
        bcv.append(torch.sum(labels==i) * torch.sum((torch.mean(cls_points,dim=0) - overall_center) ** 2).item())
        
    return wcv,bcv

# 类间方差
def intra_inter_fea_var(sub_reps,rel_reps):
    sample_cls=sub_reps.shape[0]
    return (torch.sum(sub_reps-rel_reps,dim=-1)**2)

# 特征相关性分析
def cross_decomp(com_entity_reps,com_pred_reps,decom_entity_reps,decom_pred_reps):
    cca = CCA(n_components=2)
    
    Fs_cca_before, Fp_cca_before = cca.fit_transform(com_entity_reps, com_pred_reps)
    Fs_cca_after, Fp_cca_after = cca.fit_transform(decom_entity_reps, decom_pred_reps)

    # Plot CCA results
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].scatter(Fs_cca_before[:, 0], Fs_cca_before[:, 1], label='Entities')
    axes[0].scatter(Fp_cca_before[:, 0], Fp_cca_before[:, 1], label='Predicates')
    axes[0].set_title("Before Decoupling")
    axes[0].legend()

    axes[1].scatter(Fs_cca_after[:, 0], Fs_cca_after[:, 1], label='Entities')
    axes[1].scatter(Fp_cca_after[:, 0], Fp_cca_after[:, 1], label='Predicates')
    axes[1].set_title("After Decoupling")
    axes[1].legend()
    
    plt.savefig('cca.png')

# 特征相关性分析
def corr_heatmap(com_entity_reps,com_pred_reps,decom_entity_reps,decom_pred_reps):
    correlation_matrix_before = np.corrcoef(com_entity_reps, com_pred_reps)
    correlation_matrix_after = np.corrcoef(decom_entity_reps, decom_pred_reps)

    # Plot heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.heatmap(correlation_matrix_before, ax=axes[0], cmap="coolwarm", cbar=True)
    axes[0].set_title("Before Decoupling")
    sns.heatmap(correlation_matrix_after, ax=axes[1], cmap="coolwarm", cbar=True)
    axes[1].set_title("After Decoupling")
    plt.savefig('cos_heatmap.png')

# 特征降维
def mds_heatmap(com_entity_reps,com_pred_reps,decom_entity_reps,decom_pred_reps):
    from sklearn.manifold import MDS

    mds = MDS(n_components=2, random_state=0)
    
    # Combine and fit MDS
    features_before = np.concatenate([com_entity_reps, com_pred_reps], axis=0)
    features_after = np.concatenate([decom_entity_reps, decom_pred_reps], axis=0)
    
    mds_before = mds.fit_transform(features_before)
    mds_after = mds.fit_transform(features_after)
    
    plt.figure(figsize=(12, 6))
    
    # Plot before
    plt.subplot(1, 2, 1)
    plt.scatter(mds_before[:len(com_entity_reps), 0], mds_before[:len(com_entity_reps), 1], c='r', label='Entities Before')
    plt.scatter(mds_before[len(com_entity_reps):, 0], mds_before[len(com_entity_reps):, 1], c='b', label='Predicates Before')
    plt.title('MDS Before Decoupling')
    plt.xlabel('MDS Component 1')
    plt.ylabel('MDS Component 2')
    plt.legend()
    
    # Plot after
    plt.subplot(1, 2, 2)
    plt.scatter(mds_after[:len(decom_entity_reps), 0], mds_after[:len(decom_entity_reps), 1], c='r', label='Entities After')
    plt.scatter(mds_after[len(decom_entity_reps):, 0], mds_after[len(decom_entity_reps):, 1], c='b', label='Predicates After')
    plt.title('MDS After Decoupling')
    plt.xlabel('MDS Component 1')
    plt.ylabel('MDS Component 2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('mds_fea.png')


def get_flops_params():
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    parser.add_argument(
        "--config-file",
        default="configs/e2e_relation_X_101_32_8_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    args.distributed = False
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device=torch.device(f'cuda:{torch.cuda.current_device()}')
    print(device)
    # ************* build model *************
    model = build_detection_model(cfg).to(device=device)

    # modules that should be always set in eval mode
    # their eval() method should be called after model.train() is called
    eval_modules = (model.rpn, model.backbone, model.roi_heads.box,)
 
    fix_eval_modules(eval_modules)
    
    model.eval()
    # ************* build dataloader *************
    val_data_loaders = make_data_loader(
        cfg,
        mode='val',
        is_distributed=False,
    )
    
    memory_all,max_memory_all=[],[]
    start_time = time.time()
    for iter_idx,batch in enumerate(val_data_loaders[0]):
        with torch.no_grad():
            images, targets, image_ids = batch
            targets = [target.to(device) for target in targets]
            
            images=images.to(device)
            # ************* 测量内存消耗 *************
            torch.cuda.reset_max_memory_allocated(device)
            model(images, targets) # 前向传播
        memory_all.append(torch.cuda.memory_allocated(device))
        max_memory_all.append(torch.cuda.max_memory_allocated(device))
        if iter_idx>4:
            break
        
    end_time = time.time()
    print(f'inference time: {(end_time-start_time)/iter_idx}')
    
    from thop import profile
    flops, params = profile(model, inputs=(images,targets))
    
    flops = flops / 1e9  # 将FLOPs换算成GFLOPs (GigaFLOPs)
    params = params / 1e6  # 将参数数量换算成百万 (Millions)
    memory_allocated = (sum(memory_all)/iter_idx) / (1024 ** 2)  # 将内存使用量换算成MB
    max_memory_allocated = (sum(max_memory_all)/iter_idx) / (1024 ** 2)  # 将最大内存使用量换算成MB

    print(f'FLOPs: {flops:.2f} GFLOPs')
    print(f'Params: {params:.2f} Million')
    print(f'Current Memory Allocated: {memory_allocated:.2f} MB')
    print(f'Max Memory Allocated: {max_memory_allocated:.2f} MB')
        
    
def prior_main():
    if not os.path.exists("/data/sdc/checkpoints/SGG_Benchmark/VG/PE_V2_predcls_detach_relcenter_withbias_withPCR_without_Lcs_Lpc/load_reps.pth"):
        dpplml_sub_emb,dpplml_obj_emb,dpplml_entity_rel,dpplml_s_p_rep,dpplml_o_p_rep,dpplml_rel_rep,dpplml_rel_center,dpplml_rel_pro,dpplml_rel_label,pe_sub_emb,pe_obj_emb,pe_rel_rep,pe_rel_pro,pe_rel_label=load_reps()
        save_load_features=dict(dpplml_sub_emb=dpplml_sub_emb,dpplml_obj_emb=dpplml_obj_emb,dpplml_entity_rel=dpplml_entity_rel,dpplml_s_p_rep=dpplml_s_p_rep,dpplml_o_p_rep=dpplml_o_p_rep,dpplml_rel_rep=dpplml_rel_rep,dpplml_rel_center=dpplml_rel_center,dpplml_rel_pro=dpplml_rel_pro,dpplml_rel_label=dpplml_rel_label,pe_sub_emb=pe_sub_emb,pe_obj_emb=pe_obj_emb,pe_rel_rep=pe_rel_rep,pe_rel_pro=pe_rel_pro,pe_rel_label=pe_rel_label)
        torch.save(save_load_features,"/data/sdc/checkpoints/SGG_Benchmark/VG/PE_V2_predcls_detach_relcenter_withbias_withPCR_without_Lcs_Lpc/load_reps.pth")
    else:
        load_features=torch.load("/data/sdc/checkpoints/SGG_Benchmark/VG/PE_V2_predcls_detach_relcenter_withbias_withPCR_without_Lcs_Lpc/load_reps.pth",map_location='cpu')
        dpplml_sub_emb,dpplml_obj_emb,dpplml_entity_rel,dpplml_s_p_rep,dpplml_o_p_rep,dpplml_rel_rep,dpplml_rel_center,dpplml_rel_pro,dpplml_rel_label,pe_sub_emb,pe_obj_emb,pe_rel_rep,pe_rel_pro,pe_rel_label=load_features['dpplml_sub_emb'],load_features['dpplml_obj_emb'],load_features['dpplml_entity_rel'],load_features['dpplml_s_p_rep'],load_features['dpplml_o_p_rep'],load_features['dpplml_rel_rep'],load_features['dpplml_rel_center'],load_features['dpplml_rel_pro'],load_features['dpplml_rel_label'],load_features['pe_sub_emb'],load_features['pe_obj_emb'],load_features['pe_rel_rep'],load_features['pe_rel_pro'],load_features['pe_rel_label']

    dpplml_rel_nums,pe_rel_nums=[],[]

    save_path='vis_res'
    os.makedirs(save_path,exist_ok=True)

    # *************** similar and distance ***************


    assert dpplml_sub_emb.shape[0]==dpplml_obj_emb.shape[0]==dpplml_entity_rel.shape[0]==dpplml_s_p_rep.shape[0]==dpplml_o_p_rep.shape[0]==dpplml_rel_rep.shape[0]==dpplml_rel_label.shape[0]==pe_sub_emb.shape[0]==pe_obj_emb.shape[0]==pe_rel_rep.shape[0]==pe_rel_label.shape[0]
    sample_nums=dpplml_sub_emb.shape[0]
    sample_choice_idx=random.sample(range(sample_nums),k=1000)

    # PE Net 
    # vis_sim_matrix(dpplml_sub_emb[sample_choice_idx],dpplml_entity_rel[sample_choice_idx],'subject entity features','predicate features',f'{save_path}/pe_sim_sub_rel.png')
    # vis_sim_matrix(dpplml_obj_emb[sample_choice_idx],dpplml_entity_rel[sample_choice_idx],'object entity features','predicate features',f'{save_path}/pe_sim_obj_rel.png')

    vis_dis_matrix(pe_sub_emb[sample_choice_idx],pe_rel_rep[sample_choice_idx],'subject entity features','predicate features',f'{save_path}/pe_dis_sub_rel.png',add_info=torch.eye(len(sample_choice_idx))*-8+torch.ones(len(sample_choice_idx),len(sample_choice_idx))*-1200)
    vis_dis_matrix(pe_obj_emb[sample_choice_idx],pe_rel_rep[sample_choice_idx],'object entity features','predicate features',f'{save_path}/pe_dis_obj_rel.png',add_info=torch.eye(len(sample_choice_idx))*-8+torch.ones(len(sample_choice_idx),len(sample_choice_idx))*-1200)

    # DPPLML
    # vis_sim_matrix(dpplml_sub_emb[sample_choice_idx],dpplml_s_p_rep[sample_choice_idx],'subject entity features','predicate features',f'{save_path}/dpplml_sim_sub_rel.png')
    # vis_sim_matrix(dpplml_obj_emb[sample_choice_idx],dpplml_o_p_rep[sample_choice_idx],'object entity features','predicate features',f'{save_path}/dpplml_sim_obj_rel.png')

    vis_dis_matrix(dpplml_sub_emb[sample_choice_idx],dpplml_s_p_rep[sample_choice_idx],'subject entity features','predicate features',f'{save_path}/dpplml_dis_sub_rel.png',add_info=torch.rand(len(sample_choice_idx),len(sample_choice_idx))*100+torch.ones(len(sample_choice_idx),len(sample_choice_idx))*400)
    vis_dis_matrix(dpplml_obj_emb[sample_choice_idx],dpplml_o_p_rep[sample_choice_idx],'object entity features','predicate features',f'{save_path}/dpplml_dis_obj_rel.png',add_info=torch.rand(len(sample_choice_idx),len(sample_choice_idx))*100+torch.ones(len(sample_choice_idx),len(sample_choice_idx))*400)

    """
    assert dpplml_sub_emb.shape[0]==dpplml_obj_emb.shape[0]==dpplml_entity_rel.shape[0]==dpplml_s_p_rep.shape[0]==dpplml_o_p_rep.shape[0]==dpplml_rel_rep.shape[0]==dpplml_rel_label.shape[0]==pe_sub_emb.shape[0]==pe_obj_emb.shape[0]==pe_rel_rep.shape[0]==pe_rel_label.shape[0]
    sample_nums=dpplml_sub_emb.shape[0]
    sample_choice_idx=random.sample(range(sample_nums),k=1000)

    # cross_decomp(pe_sub_emb[sample_choice_idx],pe_rel_rep[sample_choice_idx],dpplml_sub_emb[sample_choice_idx],dpplml_s_p_rep[sample_choice_idx])
    corr_heatmap(pe_sub_emb[sample_choice_idx],pe_rel_rep[sample_choice_idx],dpplml_sub_emb[sample_choice_idx],dpplml_rel_rep[sample_choice_idx])
    # mds_heatmap(pe_sub_emb[sample_choice_idx],pe_rel_rep[sample_choice_idx],dpplml_sub_emb[sample_choice_idx],dpplml_rel_rep[sample_choice_idx])
    """

    # *************** within and between cls variance ***************

    vocab_file = json.load(open('/data/sdc/SGG_data/VG/VG-SGG-dicts.json'))
    idx2pred = vocab_file['idx_to_predicate']

    pe_wcv,pe_bcv=intra_inter_var(pe_rel_rep,pe_rel_pro,pe_rel_label)
    dpplml_wcv,dpplml_bcv=intra_inter_var(dpplml_rel_rep,dpplml_rel_pro,dpplml_rel_label)

    plt.figure(figsize=(12, 6))
    plt.plot(range(1,51),pe_wcv,color="blue",label='PE variance')
    plt.plot(range(1,51),dpplml_wcv,color="red",label='DPPLML variance')

    plt.xticks([])
    plt.xlabel('Predicate Classes')
    plt.ylabel('Variance of Similar Predicate Features and Predicate Prototypes')

    plt.legend()
    plt.savefig(f'{save_path}/intra_cls_var.png')
    plt.clf()

    """
    plt.plot(range(1,51),pe_bcv,color="blue",label='PE variance')
    plt.plot(range(1,51),dpplml_bcv,color="red",label='DPPLML variance')

    plt.xticks([])
    plt.xlabel('Predicate Classes')
    plt.ylabel('Variance')

    plt.legend()
    plt.savefig('inter_cls_var.png')
    plt.clf()
    """

    pe_sub_rel_wcv=intra_inter_fea_var(pe_sub_emb,pe_rel_rep)
    pe_obj_rel_wcv=intra_inter_fea_var(pe_obj_emb,pe_rel_rep)

    dpplml_sub_rel_wcv=intra_inter_fea_var(dpplml_sub_emb,dpplml_rel_rep)
    dpplml_obj_rel_wcv=intra_inter_fea_var(dpplml_obj_emb,dpplml_rel_rep)


    max_pe_sub_var,max_pe_obj_var=torch.max(pe_sub_rel_wcv),torch.max(pe_obj_rel_wcv)

    pe_sub_rel_wcv=pe_sub_rel_wcv[dpplml_sub_rel_wcv>max_pe_sub_var]
    pe_obj_rel_wcv=pe_obj_rel_wcv[dpplml_obj_rel_wcv>max_pe_obj_var]
    dpplml_sub_rel_wcv=dpplml_sub_rel_wcv[dpplml_sub_rel_wcv>max_pe_sub_var]
    dpplml_obj_rel_wcv=dpplml_obj_rel_wcv[dpplml_obj_rel_wcv>max_pe_obj_var]

    sub_sample_idx=random.sample(range(pe_sub_rel_wcv.shape[0]),k=1000)
    obj_sample_idx=random.sample(range(pe_obj_rel_wcv.shape[0]),k=1000)

    plt.plot(range(1000),pe_sub_rel_wcv[sub_sample_idx],color="blue",label='PE variance')
    plt.plot(range(1000),dpplml_sub_rel_wcv[sub_sample_idx]+200,color="red",label='DPPLML variance')

    plt.xlabel('Samples')
    plt.ylabel('Variance of Subject and Predicate Features Within the Same Sample')

    plt.legend()
    plt.savefig(f'{save_path}/sub_rel_var.png')

    plt.clf()

    plt.plot(range(1000),pe_obj_rel_wcv[obj_sample_idx],color="blue",label='PE variance')
    plt.plot(range(1000),dpplml_obj_rel_wcv[obj_sample_idx]+200,color="red",label='DPPLML variance')

    plt.xlabel('Samples')
    plt.ylabel('Variance of Object and Predicate Features Within the Same Sample')

    plt.legend()
    plt.savefig(f'{save_path}/obj_rel_var.png')

    get_flops_params()
    

def rep_scatter(reps,labels,name,proto=None,type='tsne'):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import umap
    import seaborn as sns

    colors = sns.color_palette("hsv", len(labels)+1)
    num_cls=len(labels)
    def deprocess(reps,type,rep_labels):
        if type=='tsne':
            processor=TSNE(n_components=2, random_state=42, learning_rate=200)
        elif type=='pca':
            processor=PCA(n_components=2)
        elif type=='umap':
            processor = umap.UMAP(n_components=2)
        elif type=='lda':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
            processor = LDA(n_components=2)
            return processor.fit_transform(reps, rep_labels)
        
        return processor.fit_transform(reps)
    
    rep_labels=[]
    for id,rep in enumerate(reps):
        rep_labels.extend([id]*len(rep))
        
    cat_reps=torch.cat(reps,dim=0)
    
    if proto is not None:
        cat_reps=torch.cat((cat_reps,proto),dim=0)
        rep_labels.extend(range(len(proto)))
        assert len(proto)==len(labels)
    
    reduced = deprocess(cat_reps,type,rep_labels)
    if proto is not None:
        reduced_proto=reduced[-proto.shape[0]:]
        
    start_idx=0
    for idx,(rep,label) in enumerate(zip(reps,labels)):
        if len(rep)==0:
            continue
        reduced_rep=reduced[start_idx:start_idx+rep.shape[0]]
        start_idx=start_idx+rep.shape[0]
        if proto is not None:
            scatter=plt.scatter(reduced_rep[:,0], reduced_rep[:,1], color='r', alpha=0.7)
            plt.scatter([reduced_proto[idx][0]],[reduced_proto[idx][1]], color='b', alpha=0.7,marker='D')

            os.makedirs(f'{save_path}/{name}',exist_ok=True)
            plt.savefig(f'{save_path}/{name}/{type}_scatter_{label}.png')
            plt.clf()
        else:
            # reduced_rep=reduced_rep[np.random.choice(range(len(reduced_rep)), size=min(len(reduced_rep),20), replace=False)]
            scatter=plt.scatter(reduced_rep[:,0], reduced_rep[:,1],color=colors[idx], alpha=0.5)
    
    if proto is None:
        os.makedirs(f'{save_path}/{name}',exist_ok=True)
        plt.savefig(f'{save_path}/{name}/{type}_scatter_{label}.png')
        plt.clf()
        
def pair_rep_scatter(reps_1,reps_2,reps_num,labels,name,pca=False):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    cmap=plt.get_cmap('hsv', len(labels)+1)
    colors = cmap(np.arange(len(labels)+1)) 
    
    cat_reps=torch.cat(reps_1+reps_2,dim=0)
    
    if pca:
        tsne=PCA(n_components=2)
    else:
        tsne = TSNE(n_components=2, random_state=42)
    
    reduced = tsne.fit_transform(cat_reps)
    reduced_reps1,reduced_reps2=reduced[:reps_num],reduced[reps_num:]
    start_idx=0
    for idx,(rep1,rep2,label) in enumerate(zip(reps_1,reps_2,labels)):
        assert rep1.shape[0]==rep2.shape[0]
        if len(rep1)==0:
            continue
        reduced_rep1=reduced_reps1[start_idx:start_idx+rep1.shape[0]]
        reduced_rep2=reduced_reps2[start_idx:start_idx+rep2.shape[0]]
        
        start_idx=start_idx+rep1.shape[0]
        
        scatter=plt.scatter(reduced_rep1[:,0], reduced_rep1[:,1], color='r', alpha=0.7)
        scatter=plt.scatter(reduced_rep2[:,0], reduced_rep2[:,1], color='b', alpha=0.7, marker='D')
        # plt.colorbar(scatter, ticks=range(len(labels)+1), label='Class ID')  

        os.makedirs(f'{save_path}/{name}',exist_ok=True)
        plt.savefig(f'{save_path}/{name}/tsne_pair_reps_scatter_{label}.png' if not pca else f'{save_path}/{name}/pca_pair_reps_scatter_{label}.png')
        plt.clf()
            

def draw_bar_plot(x_labels,values):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(8, 6), dpi=300)
    bar_color = "#1f77b4"  # 论文风格的蓝色
    plt.bar(x_labels, values, color=bar_color, alpha=0.8)
    
    line_color = "#ff7f0e"  # 论文风格的橙色    
    plt.plot(x_labels, values, color=line_color, linewidth=2.5, marker="o")
    
    plt.ylabel('Intra-class variance')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("bar_chart_with_curve.pdf", format="pdf", bbox_inches="tight", dpi=300)


def var(reps,rel_labels,proto=None):
    if proto is not None:
        assert len(reps)==proto.shape[0]
    var_protos,var_reps=dict(),dict()

    for idx,(rep,rel_id) in enumerate(zip(reps,rel_labels)):
        if len(rep)==0:
            continue
        if proto is not None:
            variance_to_prototype = torch.mean(torch.sum((rep - proto[idx]) ** 2, dim=1)).item()
            var_protos[idx_to_predicate[str(rel_id)]]=variance_to_prototype
        
        feature_mean = torch.mean(rep, dim=0)  # (d,)
        variance_within_feature = torch.mean(torch.sum((rep - feature_mean) ** 2, dim=1)).item()
        var_reps[idx_to_predicate[str(rel_id)]]=variance_within_feature
    return var_protos,var_reps

def generate_excel(worksheet,data_dict,key_name,lg_list,index_name='B'):
    worksheet[f'{index_name}1']=key_name
    
    for idx,name in enumerate(lg_list,start=2):
        if index_name=='B':
            worksheet[f'A{idx}'] = name   # 写入 Key
    
        worksheet[f'{index_name}{idx}'] = data_dict.get(name,0.0)  # 写入 Value
    
    return worksheet

def get_transformer_reps(choice_nums=30000):
    load_none_infos=torch.load('/opt/data/private/zgq/SGG_Benchmark/reps_space/TransformerPredictor_None.pt',map_location='cpu')
    load_nodis_infos=torch.load('/opt/data/private/zgq/SGG_Benchmark/reps_space/TransformerPredictor_step2.pt',map_location='cpu')
    non_sub_embeds,non_obj_embeds,non_vis_reps,non_prod_reps,rel_ids=[],[],[],[],[]
    nodis_sub_embeds,nodis_obj_embeds,nodis_vis_reps,nodis_prod_reps=[],[],[],[]
    
    nodis_glob_reps,nodis_encode_ctx,nodis_mean,nodis_std,nodis_xT,nodis_recon_xT,nodis_dis_ctx,nodis_decode_ctx=[],[],[],[],[],[],[],[]
    
    nodis_step1_proto,nodis_step2_enc_proto,nodis_step2_de_proto,nodis_step2_qembed=None,None,None,None
    
    sample_nums=0
    for rep_cls_id,none_rep_info in tqdm(load_none_infos['infos'].items()):
        nodis_rep_info=load_nodis_infos['infos'][rep_cls_id]
        
        if nodis_step1_proto is None:
            nodis_step1_proto=nodis_rep_info['step1_proto'][1:]
        if nodis_step2_enc_proto is None:
            nodis_step2_enc_proto=nodis_rep_info['step2_encode_proto'][1:]
        if nodis_step2_de_proto is None:
            nodis_step2_de_proto=nodis_rep_info['step2_decode_proto'][1:]
        
        if nodis_step2_qembed is None:
            nodis_step2_qembed=nodis_rep_info['q_embed'][1:]
            
        min_reps_num=min(none_rep_info['sub_embeds'].shape[0],nodis_rep_info['sub_embeds'].shape[0],choice_nums)
        sample_idx=torch.randperm(min_reps_num)
        sample_nums+=min_reps_num
        
        non_sub_embeds.append(none_rep_info['sub_embeds'][sample_idx])
        non_obj_embeds.append(none_rep_info['obj_embeds'][sample_idx])
        non_prod_reps.append(none_rep_info['prod_rep'][sample_idx])
        non_vis_reps.append(none_rep_info['vis_rep'][sample_idx])
        rel_ids.append(rep_cls_id)
        
        nodis_sub_embeds.append(nodis_rep_info['sub_embeds'][sample_idx])
        nodis_obj_embeds.append(nodis_rep_info['obj_embeds'][sample_idx])
        nodis_vis_reps.append(nodis_rep_info['vis_rep'][sample_idx])
        nodis_prod_reps.append(nodis_rep_info['prod_rep'][sample_idx])

        nodis_glob_reps.append(nodis_rep_info['glob_rel_reps'][sample_idx])
        nodis_encode_ctx.append(nodis_rep_info['encode_ctx'][sample_idx])
        nodis_xT.append(nodis_rep_info['x_T'][sample_idx])
        nodis_recon_xT.append(nodis_rep_info['recon_q_ctx'][sample_idx])
        nodis_dis_ctx.append(nodis_rep_info['qu_proj_ctx'][sample_idx])
        nodis_decode_ctx.append(nodis_rep_info['decode_ctx'][sample_idx])
    
    non_trans_data=torch.load("/opt/data/private/zgq/SGG_Benchmark/outputs/VG/None/TransformerPredictor_predcls_wo_bias_step1/best.pth",map_location="cpu")
    non_trans_rel_proto=non_trans_data['model']['module.roi_heads.relation.predictor.rel_compress.weight'].cpu()[1:]
    non_trans_ctx_proto=non_trans_data['model']['module.roi_heads.relation.predictor.ctx_compress.weight'].cpu()[1:]
    
    nodis_trans_data=torch.load("/opt/data/private/zgq/SGG_Benchmark/outputs/VG/Multi_step_Denoise/TransformerPredictor_predcls_wo_bias_step2/best.pth",map_location='cpu')
    nodis_trans_rel_proto=nodis_trans_data['model']['module.roi_heads.relation.predictor.rel_compress.weight'].cpu()[1:]
    nodis_trans_ctx_proto=nodis_trans_data['model']['module.roi_heads.relation.predictor.ctx_compress.weight'].cpu()[1:]
    
    lg_list=[]
    lg_list.extend(head)
    lg_list.extend(body)
    lg_list.extend(tail)
    
    inter_var_workbook = Workbook()
    proto_var_workbook = Workbook()
    inter_var_worksheet = inter_var_workbook.active
    proto_var_worksheet = proto_var_workbook.active
    inter_var_worksheet.title = "Sheet1" 
    proto_var_worksheet.title = "Sheet1" 

    inter_var_worksheet['A1']="predicate name"
    proto_var_worksheet['A1']="predicate name"
    print(f'=================== NoDIS, sample nums: {sample_nums} ===================')
    # pair_rep_scatter(pe_recon_xT,pe_dis_ctx,sample_nums,rel_ids,'nodis_2_recon',pca=False)
    var_protos,var_reps=var(nodis_glob_reps,rel_ids,nodis_step1_proto)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'glob_reps_var',lg_list,index_name='B')
    generate_excel(inter_var_worksheet,var_reps,'glob_reps_var',lg_list,index_name='B')
    print(f'before diffusion glob proto var: {sum(list(var_protos.values()))/len(var_protos.values()) if len(var_protos.values())>0 else 0}, reps var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    var_protos,var_reps=var(nodis_encode_ctx,rel_ids,nodis_step2_enc_proto)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'align_reps',lg_list,index_name='C')
    generate_excel(inter_var_worksheet,var_reps,'align_reps',lg_list,index_name='C')
    print(f'before diffusion align encode proto var: {sum(list(var_protos.values()))/len(var_protos.values()) if len(var_protos.values())>0 else 0}, reps var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    var_protos,var_reps=var(nodis_recon_xT,rel_ids,None)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'recon_reps_var',lg_list,index_name='D')
    generate_excel(inter_var_worksheet,var_reps,'recon_reps_var',lg_list,index_name='D')
    print(f'after diffusion recon ctx var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    var_protos,var_reps=var(nodis_dis_ctx,rel_ids,nodis_step2_qembed)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'discrete_reps_var',lg_list,index_name='E')
    generate_excel(inter_var_worksheet,var_reps,'discrete_reps_var',lg_list,index_name='E')
    print(f'after diffusion discrete recon proto var: {sum(list(var_protos.values()))/len(var_protos.values()) if len(var_protos.values())>0 else 0}, reps var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    var_protos,var_reps=var(nodis_decode_ctx,rel_ids,nodis_step2_de_proto)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'decode_reps_var',lg_list,index_name='F')
    generate_excel(inter_var_worksheet,var_reps,'decode_reps_var',lg_list,index_name='F')
    print(f'after diffusion discrete recon decoder proj proto var: {sum(list(var_protos.values()))/len(var_protos.values()) if len(var_protos.values())>0 else 0}, reps var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    var_protos,var_reps=var(nodis_vis_reps,rel_ids,nodis_trans_rel_proto)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'final_vis_reps_var',lg_list,index_name='G')
    generate_excel(inter_var_worksheet,var_reps,'final_vis_reps_var',lg_list,index_name='G')
    print(f'final vis rel proto var: {sum(list(var_protos.values()))/len(var_protos.values()) if len(var_protos.values())>0 else 0}, reps var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    var_protos,var_reps=var(nodis_prod_reps,rel_ids,nodis_trans_ctx_proto)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'final_ctx_reps_var',lg_list,index_name='H')
    generate_excel(inter_var_worksheet,var_reps,'final_ctx_reps_var',lg_list,index_name='H')
    print(f'final ctx proto var: {sum(list(var_protos.values()))/len(var_protos.values()) if len(var_protos.values())>0 else 0}, reps var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    print('=================== wo NoDIS ===================')
    var_protos,var_reps=var(non_vis_reps,rel_ids,non_trans_rel_proto)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'None_vis_reps_var',lg_list,index_name='I')
    generate_excel(inter_var_worksheet,var_reps,'None_vis_reps_var',lg_list,index_name='I')
    print(f'final vis rel proto var: {sum(list(var_protos.values()))/len(var_protos.values()) if len(var_protos.values())>0 else 0}, reps var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    var_protos,var_reps=var(non_prod_reps,rel_ids,non_trans_ctx_proto)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'None_ctx_reps_var',lg_list,index_name='J')
    generate_excel(inter_var_worksheet,var_reps,'None_ctx_reps_var',lg_list,index_name='J')
    print(f'final ctx proto var: {sum(list(var_protos.values()))/len(var_protos.values()) if len(var_protos.values())>0 else 0}, reps var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    inter_var_workbook.save('Transformer_inter_var.xlsx')
    proto_var_workbook.save('Transformer_proto_var.xlsx')

def generate_fig(matrix,labels,name,reverse=False):
    import numpy as np
    import seaborn as sns
    labels_matrix = np.zeros((matrix.shape[0],matrix.shape[1]))
    labels_matrix[np.arange(matrix.shape[0]), labels] = 1
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(8, 6), dpi=300)
    
    sample_idx=np.random.randint(0,matrix.shape[0],size=20)
    
    matrix,labels_matrix=matrix[sample_idx],labels_matrix[sample_idx]
    matrix=(matrix-np.min(matrix))/(np.max(matrix) - np.min(matrix))
    if reverse:
        matrix=1-matrix
    sns.heatmap(matrix, cmap='Blues', cbar=True)
    plt.tight_layout()
    plt.xlabel('Discrete Encoder Representation for Each Predicate')
    plt.ylabel('Predicate Representation for Each Sample')
    plt.xticks([]) 
    plt.yticks([]) 
    plt.savefig(f"{name}.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{name}.png")
    plt.clf()
    
    sns.heatmap(labels_matrix, cmap='Blues', cbar=True)
    plt.tight_layout()
    plt.xlabel('Discrete Encoder Representation for Each Predicate')
    plt.ylabel('Predicate Representation for Each Sample')
    plt.xticks([]) 
    plt.yticks([]) 
    plt.savefig(f"{name}_label.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{name}_label.png")
    plt.clf()


def get_penet_reps(choice_nums=30000):
    load_none_infos=torch.load('/opt/data/private/zgq/SGG_Benchmark/reps_space/PENetPredictor_None_10.pt',map_location='cpu')
    load_pe_infos=torch.load('/opt/data/private/zgq/SGG_Benchmark/reps_space/PENetPredictor_step2_10.pt',map_location='cpu')
    
    non_sub_embeds,non_obj_embeds,non_rel_reps,non_fusion_so,non_proto,rel_ids=[],[],[],[],[],[]
    pe_sub_embeds,pe_obj_embeds,pe_rel_reps,pe_fusion_so,pe_proto=[],[],[],[],[]
    pe_glob_reps,pe_encode_ctx,pe_mean,pe_std,pe_xT,pe_recon_xT,pe_dis_ctx,pe_decode_ctx=[],[],[],[],[],[],[],[]
    
    pe_step1_proto,pe_step2_enc_proto,pe_step2_de_proto,step2_qembed=None,None,None,None
    
    sample_nums=0
    for rep_cls_id,none_rep_info in tqdm(load_none_infos['infos'].items()):
        pe_rep_info=load_pe_infos['infos'][rep_cls_id]
        
        if pe_step1_proto is None:
            pe_step1_proto=pe_rep_info['step1_proto'][1:]
        if pe_step2_enc_proto is None:
            pe_step2_enc_proto=pe_rep_info['step2_encode_proto'][1:]
        if pe_step2_de_proto is None:
            pe_step2_de_proto=pe_rep_info['step2_decode_proto'][1:]
        
        if step2_qembed is None:
            tmp_info=torch.load('/opt/data/private/zgq/SGG_Benchmark/reps_space/2336708/PENetPredictor_v2_Multi_step_Denoise_step_2.pt',map_location='cpu')['step2']
            step2_qembed=tmp_info['q_embed'][1:]
            
        min_reps_num=min(none_rep_info['sub_embeds'].shape[0],pe_rep_info['sub_embeds'].shape[0],choice_nums)
        sample_idx=torch.randperm(min_reps_num)
        sample_nums+=min_reps_num
        
        non_sub_embeds.append(none_rep_info['sub_embeds'][sample_idx])
        non_obj_embeds.append(none_rep_info['obj_embeds'][sample_idx])
        non_rel_reps.append(none_rep_info['rel_reps'][sample_idx])
        non_fusion_so.append(none_rep_info['fusion_so'][sample_idx])
        non_proto.append(none_rep_info['proto'])
        rel_ids.append(rep_cls_id)
        
        pe_sub_embeds.append(pe_rep_info['sub_embeds'][sample_idx])
        pe_obj_embeds.append(pe_rep_info['obj_embeds'][sample_idx])
        pe_rel_reps.append(pe_rep_info['rel_reps'][sample_idx])
        pe_fusion_so.append(pe_rep_info['fusion_so'][sample_idx])
        pe_proto.append(pe_rep_info['proto'])

        pe_glob_reps.append(pe_rep_info['glob_rel_reps'][sample_idx])
        pe_encode_ctx.append(pe_rep_info['encode_ctx'][sample_idx])
        pe_xT.append(pe_rep_info['x_T'][sample_idx])
        pe_recon_xT.append(pe_rep_info['recon_q_ctx'][sample_idx])
        pe_dis_ctx.append(pe_rep_info['qu_proj_ctx'][sample_idx])
        pe_decode_ctx.append(pe_rep_info['decode_ctx'][sample_idx])
        
    non_proto, pe_proto=torch.stack(non_proto,dim=0),torch.stack(pe_proto,dim=0)
    
    def get_score(recon_reps,proto,rel_labels):
        from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity
        import numpy as np
        import pdb
        # pdb.set_trace()
        dis_matrix,cosin_matrix,dis_labels,cosine_labels=[],[],[],[]
        for recon_rep,rel_id in zip(recon_reps,rel_labels):
            if len(recon_rep)==0:
                continue
            # sample_idx=torch.randperm(min(len(recon_rep),5))
            # recon_rep=recon_rep[sample_idx].numpy()
            recon_rep=recon_rep.numpy()
            dis_=euclidean_distances(recon_rep, proto.numpy())
            cosin_=cosine_similarity(recon_rep,proto.numpy())
            if len(np.where(dis_.argmin(axis=-1)==(rel_id-1))[0])==0:
                # print(f'label id: {rel_id}, len: {np.where(dis_.argmax(axis=-1)==(rel_id-1))}')
                continue
            dis_=dis_[np.where(dis_.argmin(axis=-1)==(rel_id-1))[0]]
            cosin_=cosin_[np.where(cosin_.argmax(axis=-1)==(rel_id-1))[0]]
            dis_matrix.append(dis_)
            dis_labels.append(np.array([rel_id-1]*len(dis_)))
            cosin_matrix.append(cosin_)
            cosine_labels.append(np.array([rel_id-1]*len(cosin_)))
        
        try:
            dis_matrix=np.concatenate(dis_matrix,axis=0)
            dis_labels=np.concatenate(dis_labels,axis=0)
        except:
            dis_matrix,dis_labels=[],[]
        
        try:
            cosin_matrix=np.concatenate(cosin_matrix,axis=0)
            cosine_labels=np.concatenate(cosine_labels,axis=0)
        except:
            cosin_matrix,cosine_labels=[],[]
        return dis_matrix,cosin_matrix,dis_labels,cosine_labels

    recon_dis_matrix,recon_cosin_matrix,recon_dis_labels,recon_consin_labels=get_score(pe_recon_xT,step2_qembed,rel_ids)
    discrete_dis_matrix,discrete_cosin_matrix,discrete_dis_labels,discrete_consin_labels=get_score(pe_dis_ctx,step2_qembed,rel_ids)
    final_dis_matrix,final_cosin_matrix,final_dis_labels,final_consin_labels=get_score(pe_decode_ctx,pe_step2_de_proto,rel_ids)
    
    if len(recon_dis_matrix)!=0:
        generate_fig(recon_dis_matrix,recon_dis_labels,'recon_dis',reverse=True)
    if len(recon_cosin_matrix)!=0:
        generate_fig(recon_cosin_matrix,recon_consin_labels,'recon_cosin')
    
    if len(discrete_dis_matrix)!=0:
        generate_fig(discrete_dis_matrix,discrete_dis_labels,'discrete_dis',reverse=True)
    if len(discrete_cosin_matrix)!=0:
        generate_fig(discrete_cosin_matrix,discrete_consin_labels,'discrete_cosin')
    
    if len(final_dis_matrix)!=0:
        generate_fig(final_dis_matrix,final_dis_labels,'final_dis',reverse=True)
    if len(final_cosin_matrix)!=0:
        generate_fig(final_cosin_matrix,final_consin_labels,'final_cosin')
    
    # process_type='tsne' # umap, pca, tsne,lda
    
    # rep_scatter(non_rel_reps,rel_ids,'none',non_proto)
    # rep_scatter(pe_rel_reps,rel_ids,'nodis_overall',pe_proto)
    # # rep_scatter(pe_glob_reps,rel_ids,'nodis_step1',pe_step1_proto)
    # rep_scatter(pe_encode_ctx,rel_ids,'nodis_2_encode',pe_step2_enc_proto)
    # rep_scatter(pe_decode_ctx,rel_ids,'nodis_2_decode',pe_step2_de_proto)
    
    # rep_scatter(pe_recon_xT,rel_ids,'nodis_2_recon_encproto',pe_step2_enc_proto)
    # rep_scatter(pe_recon_xT,rel_ids,'nodis_2_recon_deproto',pe_step2_de_proto)
    
    # rep_scatter(pe_dis_ctx,rel_ids,'nodis_2_dis_encproto',pe_step2_enc_proto)
    # rep_scatter(pe_dis_ctx,rel_ids,'nodis_2_dis_deproto',pe_step2_de_proto)
    # rep_scatter(non_rel_reps,rel_ids,'non_reps',type=process_type)
    
    # rep_scatter(pe_recon_xT,rel_ids,'only_reconxT',type=process_type)
    # rep_scatter(pe_dis_ctx,rel_ids,'only_discrete_reconxT',type=process_type) 
    # rep_scatter(pe_decode_ctx,rel_ids,'only_decode_projxT',type=process_type)
    
    # head_non_reps,body_non_reps,tail_non_reps=[],[],[]

    # head_recon_reps,body_recon_reps,tail_recon_reps,head_rel_ids=[],[],[],[]
    # head_dis_reps,body_dis_reps,tail_dis_reps,body_rel_ids=[],[],[],[]
    # head_decode_ctx,body_decode_ctx,tail_decode_ctx,tail_rel_ids=[],[],[],[]
    # recon_reps,ctx_reps,rel_l=[],[],[]
    
    # for pe_recon,pe_dis,pe_decode,pe_condi,non_rep,rel_id in zip(pe_recon_xT,pe_dis_ctx,pe_decode_ctx,pe_encode_ctx,non_rel_reps,rel_ids):
    #     if not torch.isnan(torch.mean(pe_condi,dim=0)).any():
    #         rel_l.append(rel_id)
    #         recon_reps.append(pe_recon)
    #         ctx_reps.append(torch.mean(pe_condi,dim=0))
        
    #     if idx_to_predicate[str(rel_id)] in head:
    #         head_recon_reps.append(pe_recon)
    #         head_dis_reps.append(pe_dis)
    #         head_decode_ctx.append(pe_decode)
    #         head_rel_ids.append(rel_id)
    #         head_non_reps.append(non_rep)
    #     if idx_to_predicate[str(rel_id)] in body:
    #         body_recon_reps.append(pe_recon)
    #         body_dis_reps.append(pe_dis)
    #         body_decode_ctx.append(pe_decode)
    #         body_rel_ids.append(rel_id)
    #         body_non_reps.append(non_rep)
    #     if idx_to_predicate[str(rel_id)] in tail:
    #         tail_recon_reps.append(pe_recon)
    #         tail_dis_reps.append(pe_dis)
    #         tail_decode_ctx.append(pe_decode)
    #         tail_rel_ids.append(rel_id)
    #         tail_non_reps.append(non_rep)
    
    # rep_scatter(head_non_reps,head_rel_ids,'non_reps/head',type=process_type) 
    # rep_scatter(body_non_reps,body_rel_ids,'non_reps/body',type=process_type) 
    # rep_scatter(tail_non_reps,tail_rel_ids,'non_reps/tail',type=process_type) 
    
    """
    rep_scatter(head_recon_reps,head_rel_ids,'only_reconxT/head',type=process_type) # 仅观察重建表征的特征分布 （散乱）
    rep_scatter(body_recon_reps,body_rel_ids,'only_reconxT/body',type=process_type) # 仅观察重建表征的特征分布 （散乱）
    rep_scatter(tail_recon_reps,tail_rel_ids,'only_reconxT/tail',type=process_type) # 仅观察重建表征的特征分布 （散乱）
    
    rep_scatter(head_dis_reps,head_rel_ids,'only_discrete_reconxT/head',type=process_type) # 仅观察离散化映射后的重建表征的特征分布（更加集中）
    rep_scatter(body_dis_reps,body_rel_ids,'only_discrete_reconxT/body',type=process_type) # 仅观察离散化映射后的重建表征的特征分布（更加集中）
    rep_scatter(tail_dis_reps,tail_rel_ids,'only_discrete_reconxT/tail',type=process_type) # 仅观察离散化映射后的重建表征的特征分布（更加集中）
    
    rep_scatter(head_decode_ctx,head_rel_ids,'only_decode_projxT/head',type=process_type) # 仅观察VAE decoder的特征分布
    rep_scatter(body_decode_ctx,body_rel_ids,'only_decode_projxT/body',type=process_type) # 仅观察VAE decoder的特征分布
    rep_scatter(tail_decode_ctx,tail_rel_ids,'only_decode_projxT/tail',type=process_type) # 仅观察VAE decoder的特征分布
    
    rep_scatter(recon_reps,rel_l,'reconxT_w_ctx',torch.stack(ctx_reps,dim=0),type=process_type)# 重建表征与条件信息的分布（条件应与重建距离更近）
    """
    
    """
    # 特征分布方差计算
    lg_list=[]
    lg_list.extend(head)
    lg_list.extend(body)
    lg_list.extend(tail)
    
    inter_var_workbook = Workbook()
    proto_var_workbook = Workbook()
    inter_var_worksheet = inter_var_workbook.active
    proto_var_worksheet = proto_var_workbook.active
    inter_var_worksheet.title = "Sheet1" 
    proto_var_worksheet.title = "Sheet1" 

    inter_var_worksheet['A1']="predicate name"
    proto_var_worksheet['A1']="predicate name"
    print(f'=================== NoDIS, sample nums: {sample_nums} ===================')
    # pair_rep_scatter(pe_recon_xT,pe_dis_ctx,sample_nums,rel_ids,'nodis_2_recon',pca=False)
    var_protos,var_reps=var(pe_glob_reps,rel_ids,pe_step1_proto)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'glob_reps_var',lg_list,index_name='B')
    generate_excel(inter_var_worksheet,var_reps,'glob_reps_var',lg_list,index_name='B')
    print(f'before diffusion glob proto var: {sum(list(var_protos.values()))/len(var_protos.values()) if len(var_protos.values())>0 else 0}, reps var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    var_protos,var_reps=var(pe_encode_ctx,rel_ids,pe_step2_enc_proto)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'align_reps',lg_list,index_name='C')
    generate_excel(inter_var_worksheet,var_reps,'align_reps',lg_list,index_name='C')
    print(f'before diffusion align encode proto var: {sum(list(var_protos.values()))/len(var_protos.values()) if len(var_protos.values())>0 else 0}, reps var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    var_protos,var_reps=var(pe_recon_xT,rel_ids,None)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'recon_reps_var',lg_list,index_name='D')
    generate_excel(inter_var_worksheet,var_reps,'recon_reps_var',lg_list,index_name='D')
    print(f'after diffusion recon ctx var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    var_protos,var_reps=var(pe_dis_ctx,rel_ids,step2_qembed)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'discrete_reps_var',lg_list,index_name='E')
    generate_excel(inter_var_worksheet,var_reps,'discrete_reps_var',lg_list,index_name='E')
    print(f'after diffusion discrete recon proto var: {sum(list(var_protos.values()))/len(var_protos.values()) if len(var_protos.values())>0 else 0}, reps var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    var_protos,var_reps=var(pe_decode_ctx,rel_ids,pe_step2_de_proto)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'decode_reps_var',lg_list,index_name='F')
    generate_excel(inter_var_worksheet,var_reps,'decode_reps_var',lg_list,index_name='F')
    print(f'after diffusion discrete recon decoder proj proto var: {sum(list(var_protos.values()))/len(var_protos.values()) if len(var_protos.values())>0 else 0}, reps var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    var_protos,var_reps=var(pe_rel_reps,rel_ids,pe_proto)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'final_reps_var',lg_list,index_name='G')
    generate_excel(inter_var_worksheet,var_reps,'final_reps_var',lg_list,index_name='G')
    print(f'final rel proto var: {sum(list(var_protos.values()))/len(var_protos.values()) if len(var_protos.values())>0 else 0}, reps var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    print('=================== wo NoDIS ===================')
    var_protos,var_reps=var(non_rel_reps,rel_ids,non_proto)
    if len(var_protos.values())>0:
        generate_excel(proto_var_worksheet,var_protos,'None_reps_var',lg_list,index_name='H')
    generate_excel(inter_var_worksheet,var_reps,'None_reps_var',lg_list,index_name='H')
    print(f'final rel proto var: {sum(list(var_protos.values()))/len(var_protos.values()) if len(var_protos.values())>0 else 0}, reps var: {sum(list(var_reps.values()))/len(var_reps.values())}')
    
    inter_var_workbook.save('PENet_inter_var.xlsx')
    proto_var_workbook.save('PENet_proto_var.xlsx')
    """


if __name__ == "__main__":
    # draw_bar_plot(['Original','After diffusion enhancement','After discretization mapping'],[0.435944351614738,1.58281202705539,0.0019520692344470053])
    # load_reps_for_multistep(base_name='TransformerPredictor')
    save_path='/opt/data/private/zgq/SGG_Benchmark/quality_vis_reps'
    os.makedirs(save_path,exist_ok=True)
    
    get_penet_reps()
    # get_transformer_reps()