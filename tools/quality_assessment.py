# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
import os,sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.abspath(os.path.join(current_dir,'../')))

import argparse
import time
import torch

from tools.relation_train_net import fix_eval_modules
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.comm import synchronize
import numpy as np
import random
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import CCA

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

def intra_inter_var(reps,rel_pros,labels,cls_num=51):
    wcv,bcv=[],[]
    overall_center = torch.mean(reps, dim=0)
    for i in range(1,cls_num):
        cls_points=reps[labels==i]
        cls_center = rel_pros[i]
        wcv.append(torch.sum((cls_points - cls_center) ** 2).item()/len(cls_points))
        
        bcv.append(torch.sum(labels==i) * torch.sum((torch.mean(cls_points,dim=0) - overall_center) ** 2).item())
        
    return wcv,bcv

def intra_inter_fea_var(sub_reps,rel_reps):
    sample_cls=sub_reps.shape[0]
    return (torch.sum(sub_reps-rel_reps,dim=-1)**2)

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
        
    
if __name__ == "__main__":
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