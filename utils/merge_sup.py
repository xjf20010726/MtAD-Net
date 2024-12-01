import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from skimage import color,measure,graph
from scipy.spatial.distance import cdist
torch.set_printoptions(threshold=np.inf)
import math
# def merge_sup(num_sup,result_sup,y_pred,beta,c,feature_map,X):
#     properties=measure.regionprops(result_sup)
#     rag=graph.rag_mean_color(result_sup, result_sup)
#     # result_sup=torch.from_numpy(result_sup).cuda()
#     if num_sup!=len(properties):
#         num_sup=len(properties)
#     A=torch.zeros(num_sup,num_sup,requires_grad=False).cuda(device=X.device)
#     rag.nodes
#     for i in rag.edges:
#         # index_i=properties[i[0]-1].centroid
#         # index_j=properties[i[1]-1].centroid
#         index_i=np.argwhere(result_sup==properties[i[0]-1].label)
#         index_j=np.argwhere(result_sup==properties[i[1]-1].label)
#         # Ii=X[:,0,index_i[:,0],index_i[:,1]].mean()
#         # Ij=X[:,0,index_j[:,0],index_j[:,1]].mean()
#         # distances = cdist(index_i,index_j, 'euclidean')
#         # D=np.argwhere(distances==np.min(distances))
#         # A_ij=1-2*torch.sigmoid(-(torch.norm(feature_map[:,:,int(index_i[0]),int(index_i[1])]-feature_map[:,:,int(index_j[0]),int(index_j[1])],p=2)))
#         A_ij=1-2*torch.sigmoid(-(torch.norm(torch.mean(feature_map[:,:,index_i[:,0],index_i[:,1]],dim=2)-torch.mean(feature_map[:,:,index_j[:,0],index_j[:,1]],dim=2),p=2)))
#         A[i[0]-1,i[1]-1]=A_ij
#         A[i[1]-1,i[0]-1]=A_ij
#     for i in rag.edges:
#         # index_i=np.argwhere(result_sup==properties[i[0]-1].label)
#         index_j=np.argwhere(result_sup==properties[i[1]-1].label)
#         if(A[i[0]-1,i[1]-1]<beta):
#             result_sup[index_j[:,0],index_j[:,1]]=properties[i[0]-1].label
#             properties[i[1]-1].label=properties[i[0]-1].label
    
#     # for i in range(num_sup):
#     #     for j in range(i,num_sup):
#     #         if(A[i,j]<=beta):
#     #             index=np.argwhere((result_sup==properties[j].label))

#     #             # index=(result_sup==(j+1)).nonzero()
#     #             # print(index[0],index[1])
#     #             # print(type(result_sup))
#     #             result_sup[index[:,0],index[:,1]]=properties[i].label
#     # print(y_pred)
#     tmp_y=torch.where(torch.gt(y_pred, 0.5),torch.ones_like(y_pred), torch.zeros_like(y_pred))
#     # tmp_y=y_pred.argmax(dim=1)
#     tmp_y=tmp_y.squeeze()
#     # print(tmp_y)
#     # print(tmp_y.dtype,c.dtype)
#     c1=c.clone()
#     c1=c1.long()

#     # result_sup=result_sup.cpu().numpy()
#     new_label=measure.regionprops(result_sup)
#     # print(len(new_label))
#     # result_sup=torch.from_numpy(result_sup).cuda()
#     for i in range(len(new_label)):
#         # print(new_label[i].label)
#         index=np.argwhere((result_sup==new_label[i].label))
#         # (result_sup==new_label[i].label).nonzero()
#         out,counts=torch.unique(tmp_y[index[:,0],index[:,1]],return_counts=True,sorted=True)
#         # print(out,counts)
#         mc=out[counts.argmax()]
#         tmp_y[index[:,0],index[:,1]]=mc
#         if mc>=0.5*new_label[i].area:
#             tmp_y[index[:,0],index[:,1]]=mc
#         else:
#             tmp_y[index[:,0],index[:,1]]=mc+c1.item()
    
#     return tmp_y


def weight_func(graph, src, dst,n):
    Cos=graph.nodes[dst]['mean_f']@graph.nodes[n]['mean_f'].t()
    Cos=torch.clamp(Cos, min=-1, max=1)
    diff=1-2*torch.sigmoid(-(torch.sqrt(2-2*Cos)))
    # diff=
    return {'weight': diff}
def merge_func(graph, src, dst):
    all_pix=graph.nodes[src]['pixel count']+graph.nodes[dst]['pixel count']
    graph.nodes[dst]['mean_f']=graph.nodes[dst]['mean_f']*graph.nodes[dst]['pixel count']/all_pix+graph.nodes[src]['mean_f']*graph.nodes[src]['pixel count']/all_pix
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean_f'] = F.normalize(graph.nodes[dst]['mean_f'],p=2, dim=0, eps=1e-12)    

def merge_sup(result_sup,y_pred,all_F,category_num,seg_lab,beta):
    rag=graph.rag_mean_color(result_sup, result_sup)
    if category_num>2:
        y_pred=y_pred.permute(0,2,1).view(-1,category_num)
    else:
        y_pred=y_pred.permute(0,2,1).view(-1,1)
    # print(y_pred.shape)
    # y_pred=y_pred.permute(0,2,1).view(-1,1)
    target=torch.where(torch.gt(y_pred, 0.5),torch.ones_like(y_pred), torch.zeros_like(y_pred)).squeeze()
    final_result=torch.zeros(result_sup.shape).flatten()
    # print(result_sup.shape)
    # seg_map = result_sup.flatten()
    # seg_lab = [np.where(seg_map == u_label)[0]
    #             for u_label in np.unique(seg_map)]
    # b,c,h,w=feature_map.shape
    # F1=feature_map.permute(0,2,3,1).view(-1,c)

    # _,W,H=vit_feature.shape
    # vit_F=vit_feature.view(W,H)
    # M_F=torch.zeros(num_sup,c).cuda()
    for i,inds in enumerate(seg_lab):
        # inds=inds.numpy().squeeze()
        
        # M_F=torch.mean(F1[inds,:],dim=0)
        # S_F=torch.sum(F1[inds,:],dim=0)
        # # print(vit_F[i,:].shape)
        # # print(len(inds))
        # # return 0
        # all_F=torch.concat((M_F,vit_F[i,:]),dim=0)
        
        rag.nodes[i+1]['mean_f']=all_F
        final_result[inds]=target[i]
        # M_F[i,:]=torch.mean(F1[inds,:],dim=0)
    labels2 = graph.merge_hierarchical(result_sup, rag, thresh=beta, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_func,
                                   weight_func=weight_func).flatten()
    seg_new_lab=[np.where(labels2 == u_label)[0]
                for u_label in np.unique(labels2)]
    
    
    # print(target.shape)
    # im_target = target.data.cpu().numpy()
    # print(im_target.shape)
    for i,inds in enumerate(seg_new_lab):
        u_labels, hist = torch.unique(final_result[inds], return_counts=True)
        final_result[inds]=u_labels[torch.argmax(hist)]
        # u_labels, hist = np.unique(im_target[inds], return_counts=True)
        # im_target[inds] = u_labels[np.argmax(hist)]
    # final_result=torch.from_numpy(im_target).long()
    final_result=final_result.reshape((result_sup.shape[0],result_sup.shape[1]))
    return final_result