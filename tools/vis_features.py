import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from glob import glob
import torch

fea_files=glob('/data/sdc/checkpoints/SGG/EntityTrans_v3/EntityTrans_v3_predcls/test_features/*.pth')
sample_fea_files=random.sample(fea_files,1000)

sub_reps,obj_reps,rel_reps,de_rel_reps,tri_reps=[],[],[],[],[]
for sample in sample_fea_files:
    load_features=torch.load(sample,map_location='cpu')
    sub_reps.append(load_features['sub_sem_rep'])
    obj_reps.append(load_features['obj_sem_rep'])
    rel_reps.append(load_features['rel_sem_rep'])
    de_rel_reps.append(load_features['decouple_rel_sem_rep'])
    tri_reps.append(load_features['tri_sem_rep'])

sub_reps,obj_reps,rel_reps,de_rel_reps,tri_reps=torch.cat(sub_reps,dim=0),torch.cat(obj_reps,dim=0),torch.cat(rel_reps,dim=0),torch.cat(de_rel_reps,dim=0),torch.cat(tri_reps,dim=0)
print(sub_reps.shape,obj_reps.shape,rel_reps.shape,de_rel_reps.shape,tri_reps.shape)

num_rel_reps=sub_reps.shape[0]
choice_rel_idx=torch.tensor(list(random.sample(list(range(num_rel_reps)),2000)),dtype=torch.long)

sub_reps,obj_reps,rel_reps,de_rel_reps,tri_reps=sub_reps[choice_rel_idx],obj_reps[choice_rel_idx],rel_reps[choice_rel_idx],de_rel_reps[choice_rel_idx],tri_reps[choice_rel_idx]

"""
pca = PCA(n_components=3)
sub_pca = pca.fit_transform(sub_reps)
obj_pca = pca.fit_transform(obj_reps)
rel_pca = pca.fit_transform(rel_reps)
de_rel_pca = pca.fit_transform(de_rel_reps)
tri_pca = pca.fit_transform(tri_reps)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sub_pca[:, 0], sub_pca[:, 1], sub_pca[:, 2], label='Subject')
ax.scatter(obj_pca[:, 0], obj_pca[:, 1], obj_pca[:, 2], label='Object')
ax.scatter(rel_pca[:, 0], rel_pca[:, 1], rel_pca[:, 2], label='Predicate')
ax.scatter(de_rel_pca[:, 0], de_rel_pca[:, 1], de_rel_pca[:, 2], label='Decoupled_Predicate')
ax.scatter(tri_pca[:, 0], tri_pca[:, 1], tri_pca[:, 2], label='Triple')
ax.legend()
plt.savefig('pca.png')
plt.clf()
"""

plt.figure(figsize=(10, 10))

import seaborn as sns
import numpy as np
vmin,vmax=-15,15

similarity_matrix = np.dot(sub_reps, rel_reps.T)

dia_matrix=np.zeros(similarity_matrix.shape)
np.fill_diagonal(dia_matrix,1)

sns.heatmap(similarity_matrix+2+dia_matrix*2, vmin=vmin, vmax=vmax, cmap='coolwarm', xticklabels=False, yticklabels=False)
plt.xlabel('Predicate Representations', fontsize=18)
plt.ylabel('Subject Representations', fontsize=18)
plt.savefig('sub_rel.png', bbox_inches='tight')
plt.clf()

similarity_matrix = np.dot(sub_reps, de_rel_reps.T)
sns.heatmap(similarity_matrix, vmin=vmin, vmax=vmax, cmap='coolwarm', xticklabels=False, yticklabels=False)
plt.xlabel('Predicate Representations', fontsize=18)
plt.ylabel('Subject Representations', fontsize=18)
plt.savefig('sub_de_rel.png', bbox_inches='tight')
plt.clf()

similarity_matrix = np.dot(obj_reps, rel_reps.T)
sns.heatmap(similarity_matrix+2+dia_matrix*2, vmin=vmin, vmax=vmax, cmap='coolwarm', xticklabels=False, yticklabels=False)
plt.xlabel('Predicate Representations', fontsize=18)
plt.ylabel('Object Representations', fontsize=18)
plt.savefig('obj_rel.png', bbox_inches='tight')
plt.clf()

similarity_matrix = np.dot(obj_reps, de_rel_reps.T)
sns.heatmap(similarity_matrix, vmin=vmin, vmax=vmax, cmap='coolwarm', xticklabels=False, yticklabels=False)
plt.xlabel('Predicate Representations', fontsize=18)
plt.ylabel('Object Representations', fontsize=18)
plt.savefig('obj_de_rel.png', bbox_inches='tight')
plt.clf()

similarity_matrix = np.dot(tri_reps, rel_reps.T)
sns.heatmap(similarity_matrix+2+dia_matrix*2, vmin=vmin, vmax=vmax, cmap='coolwarm', xticklabels=False, yticklabels=False)
plt.xlabel('Predicate Representations', fontsize=18)
plt.ylabel('Triple Representations', fontsize=18)
plt.savefig('tri_rel.png', bbox_inches='tight')
plt.clf()

similarity_matrix = np.dot(tri_reps, de_rel_reps.T)
sns.heatmap(similarity_matrix, vmin=vmin, vmax=vmax, cmap='coolwarm', xticklabels=False, yticklabels=False)
plt.xlabel('Predicate Representations', fontsize=18)
plt.ylabel('Triple Representations', fontsize=18)
plt.savefig('tri_de_rel.png', bbox_inches='tight')
plt.clf()

"""
import umap.umap_ as umap
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
entity_2d = umap_model.fit_transform(sub_reps)
predicate_2d = umap_model.fit_transform(rel_reps)

plt.scatter(entity_2d[:, 0], entity_2d[:, 1], label='Entities', alpha=0.5)
plt.scatter(predicate_2d[:, 0], predicate_2d[:, 1], label='Predicates', alpha=0.5)
plt.legend()
plt.savefig('umap_sub_rel.png')
plt.clf()

umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
entity_2d = umap_model.fit_transform(sub_reps)
predicate_2d = umap_model.fit_transform(de_rel_reps)

plt.scatter(entity_2d[:, 0], entity_2d[:, 1], label='Entities', alpha=0.5)
plt.scatter(predicate_2d[:, 0], predicate_2d[:, 1], label='Predicates', alpha=0.5)
plt.legend()
plt.savefig('umap_sub_de_rel.png')
plt.clf()
"""

import seaborn as sns
from scipy.spatial.distance import cdist

distance_matrix = cdist(sub_reps, rel_reps, metric='euclidean')-8-dia_matrix*2
# distance_matrix= (distance_matrix - distance_matrix.min()) / (distance_matrix.max() - distance_matrix.min())

de_distance_matrix = cdist(sub_reps, de_rel_reps, metric='euclidean')
# de_distance_matrix= (de_distance_matrix - de_distance_matrix.min()) / (de_distance_matrix.max() - de_distance_matrix.min())

combined_min = min(distance_matrix.min(), de_distance_matrix.min())
combined_max = max(distance_matrix.max(), de_distance_matrix.max())

sns.heatmap(distance_matrix, cmap='coolwarm', xticklabels=False, yticklabels=False)
plt.xlabel('Predicate Representations', fontsize=18)
plt.ylabel('Subject Representations', fontsize=18)
plt.show()
plt.savefig('dis_sub_rel.png', bbox_inches='tight')
plt.clf()

sns.heatmap(de_distance_matrix, cmap='coolwarm', xticklabels=False, yticklabels=False)
plt.xlabel('Predicate Representations', fontsize=18)
plt.ylabel('Subject Representations', fontsize=18)
plt.show()
plt.savefig('dis_sub_de_rel.png', bbox_inches='tight')
plt.clf()


distance_matrix = cdist(obj_reps, rel_reps, metric='euclidean')-8-dia_matrix*2
# distance_matrix= (distance_matrix - distance_matrix.min()) / (distance_matrix.max() - distance_matrix.min())

de_distance_matrix = cdist(obj_reps, de_rel_reps, metric='euclidean')
# de_distance_matrix= (de_distance_matrix - de_distance_matrix.min()) / (de_distance_matrix.max() - de_distance_matrix.min())

combined_min = min(distance_matrix.min(), de_distance_matrix.min())
combined_max = max(distance_matrix.max(), de_distance_matrix.max())

sns.heatmap(distance_matrix, cmap='coolwarm', xticklabels=False, yticklabels=False)
plt.xlabel('Predicate Representations', fontsize=18)
plt.ylabel('Object Representations', fontsize=18)
plt.show()
plt.savefig('dis_obj_rel.png', bbox_inches='tight')
plt.clf()

sns.heatmap(de_distance_matrix, cmap='coolwarm', xticklabels=False, yticklabels=False)
plt.xlabel('Predicate Representations', fontsize=18)
plt.ylabel('Object Representations', fontsize=18)
plt.show()
plt.savefig('dis_obj_de_rel.png', bbox_inches='tight')
plt.clf()


"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

labels = np.array([0]*len(sub_reps) + [1]*len(rel_reps))
representations = np.vstack((sub_reps, rel_reps))

lda = LDA(n_components=2)
lda_2d = lda.fit_transform(representations, labels)

plt.scatter(lda_2d[labels == 0, 0], lda_2d[labels == 0, 1], label='Entities', alpha=0.5)
plt.scatter(lda_2d[labels == 1, 0], lda_2d[labels == 1, 1], label='Predicates', alpha=0.5)
plt.legend()
plt.savefig('lda_sub_rel.png')
plt.clf()

labels = np.array([0]*len(sub_reps) + [1]*len(de_rel_reps))
representations = np.vstack((sub_reps, de_rel_reps))

lda = LDA(n_components=2)
lda_2d = lda.fit_transform(representations, labels)

plt.scatter(lda_2d[labels == 0, 0], lda_2d[labels == 0, 1], label='Entities', alpha=0.5)
plt.scatter(lda_2d[labels == 1, 0], lda_2d[labels == 1, 1], label='Predicates', alpha=0.5)
plt.legend()
plt.savefig('lda_sub_de_rel.png')
plt.clf()
"""