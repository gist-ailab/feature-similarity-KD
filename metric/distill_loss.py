import torch
import random
import torch.nn.functional as F
import torch.nn as nn



def cosine_loss(l , h):
    l = l.view(l.size(0), -1)
    h = h.view(h.size(0), -1)
    return torch.mean(1.0 - F.cosine_similarity(l, h))


def mse_loss(l, h):
    l = l.view(l.size(0), -1)
    h = h.view(h.size(0), -1)
    return F.mse_loss(l, h)


def l1_loss(l, h):
    l = l.view(l.size(0), -1)
    h = h.view(h.size(0), -1)
    return F.l1_loss(l, h)


def normalize(feat):
    # feat : (B, C, H, W)
    feat = feat.flatten(2)
    norm = torch.norm(feat, dim=[1,2]).unsqueeze(1).unsqueeze(2)
    return feat/norm, norm.unsqueeze(3)


class cross_kd(nn.Module):
    def __init__(self):
        super(cross_kd, self).__init__()
    
    def forward(self, feat_student, feat_student2, feat_teacher, feat_teacher2):
        feat_teacher = feat_teacher.detach()
        feat_teacher2 = feat_teacher2.detach()
        
        # Relation
        R_HH = self.relation(feat_teacher, feat_teacher2)
        R_HL = self.relation(feat_teacher, feat_student2)
        R_LH = self.relation(feat_student, feat_teacher2)
        R_LL = self.relation(feat_student, feat_student2)
        
        # Attention Loss
        distill_loss = (self.attention_loss(R_HH, R_HL) + self.attention_loss(R_HH, R_LH)) / 2 +\
                       self.attention_loss(R_HH, R_LL)
        return distill_loss
    

    def relation(self, x1, x2):
        x1_normalized, _ = normalize(x1)
        x2_normalized, _ = normalize(x2)
        affinity = torch.einsum('bcM,bcN->bMN', x1_normalized, x2_normalized)
        return affinity
    
    def attention_loss(self, a1, a2):
        distill_loss = torch.mean(1.0 - F.cosine_similarity(a1.flatten(1), a2.flatten(1)))
        return distill_loss
    
    


'''
From https://github.com/lenscloth/RKD/blob/master/metric/loss.py
'''
class RKD_cri(nn.Module):
	'''
	Relational Knowledge Distillation
	https://arxiv.org/pdf/1904.05068.pdf
	'''
	def __init__(self, w_dist=1., w_angle=2.):
		super(RKD_cri, self).__init__()

		self.w_dist  = w_dist
		self.w_angle = w_angle

	def forward(self, feat_s, feat_t):
		loss = self.w_dist * self.rkd_dist(feat_s, feat_t) + \
			   self.w_angle * self.rkd_angle(feat_s, feat_t)
		return loss

	def rkd_dist(self, feat_s, feat_t):
		feat_t_dist = self.pdist(feat_t, squared=False)
		mean_feat_t_dist = feat_t_dist[feat_t_dist>0].mean()
		feat_t_dist = feat_t_dist / mean_feat_t_dist

		feat_s_dist = self.pdist(feat_s, squared=False)
		mean_feat_s_dist = feat_s_dist[feat_s_dist>0].mean()
		feat_s_dist = feat_s_dist / mean_feat_s_dist

		loss = F.smooth_l1_loss(feat_s_dist, feat_t_dist)
		return loss

	def rkd_angle(self, feat_s, feat_t):
		# N x C --> N x N x C
		feat_t_vd = (feat_t.unsqueeze(0) - feat_t.unsqueeze(1))
		norm_feat_t_vd = F.normalize(feat_t_vd, p=2, dim=2)
		feat_t_angle = torch.bmm(norm_feat_t_vd, norm_feat_t_vd.transpose(1, 2)).view(-1)

		feat_s_vd = (feat_s.unsqueeze(0) - feat_s.unsqueeze(1))
		norm_feat_s_vd = F.normalize(feat_s_vd, p=2, dim=2)
		feat_s_angle = torch.bmm(norm_feat_s_vd, norm_feat_s_vd.transpose(1, 2)).view(-1)

		loss = F.smooth_l1_loss(feat_s_angle, feat_t_angle)
		return loss

	def pdist(self, feat, squared=False, eps=1e-12):
		feat_square = feat.pow(2).sum(dim=1)
		feat_prod   = torch.mm(feat, feat.t())
		feat_dist   = (feat_square.unsqueeze(0) + feat_square.unsqueeze(1) - 2 * feat_prod).clamp(min=eps)

		if not squared:
			feat_dist = feat_dist.sqrt()

		feat_dist = feat_dist.clone()
		feat_dist[range(len(feat)), range(len(feat))] = 0
		return feat_dist



class AT_cri(nn.Module):
	'''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks wia Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	'''
	def __init__(self, p=2):
		super(AT_cri, self).__init__()
		self.p = p

	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))
		return loss

	def attention_map(self, fm, eps=1e-6):
		am = torch.pow(torch.abs(fm), self.p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)
		return am
