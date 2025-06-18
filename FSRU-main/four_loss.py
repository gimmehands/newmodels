"""
An Lao
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-5
T1 = 0.05  # 0.05
T2 = 0.05  # 0.05

class SelfContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=T1):
        super(SelfContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size, batch_size, dtype=torch.bool).to(device)).float())

    def forward(self, q, k):
        q = F.normalize(q, dim=1)  # (bs, dim)  --->  (bs, dim)
        k = F.normalize(k, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([q, k], dim=0)  # (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1),
                                                representations.unsqueeze(0), dim=2)  # (2*bs, 2*bs)
        sim_qk = torch.diag(similarity_matrix, self.batch_size)  # (bs,)
        sim_kq = torch.diag(similarity_matrix, -self.batch_size)  # (bs,)

        nominator_qk = torch.exp(sim_qk / self.temperature)   # (bs,)
        negatives_qk = similarity_matrix[:self.batch_size, self.batch_size:]  # (bs, bs)
        denominator_qk = nominator_qk + torch.sum(self.negatives_mask * torch.exp(negatives_qk/self.temperature), dim=1)

        nominator_kq = torch.exp(sim_kq / self.temperature)
        negatives_kq = similarity_matrix[self.batch_size:, :self.batch_size]
        denominator_kq = nominator_kq + torch.sum(self.negatives_mask * torch.exp(negatives_kq/self.temperature), dim=1)

        loss_qk = torch.sum(-torch.log(nominator_qk / denominator_qk + eps)) / self.batch_size
        loss_kq = torch.sum(-torch.log(nominator_kq / denominator_kq + eps)) / self.batch_size
        loss = loss_qk + loss_kq

        return loss

class FullContrastiveLoss(nn.Module):
    def __init__(self, batch_size, class_counts, device='cuda', temperature=T2):
        super(FullContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.class_counts = class_counts
        self.num_classes = len(class_counts)
        self.device = device

        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.masks = self.create_masks(class_counts)

    def create_masks(self, class_counts):
        masks = []
        for count in class_counts:
            mask = (~torch.eye(count, count, dtype=torch.bool).to(self.device)).float()
            masks.append(mask)
        return masks

    def compute_loss(self, feature, label):
        loss = 0
        for i in range(self.num_classes):
            index_i = torch.nonzero(label == i).squeeze()
            ft_i = torch.index_select(feature, dim=0, index=index_i)
            similarity_matrix_i = F.cosine_similarity(ft_i.unsqueeze(1), ft_i.unsqueeze(0), dim=2)

            nominator_i = torch.sum(self.masks[i] * torch.exp(similarity_matrix_i / self.temperature), dim=1)
            denominator_i = nominator_i

            for j in range(self.num_classes):
                if i != j:
                    index_j = torch.nonzero(label == j).squeeze()
                    ft_j = torch.index_select(feature, dim=0, index=index_j)
                    similarity_matrix_i_j = F.cosine_similarity(ft_i.unsqueeze(1), ft_j.unsqueeze(0), dim=2)
                    denominator_i += torch.sum(torch.exp(similarity_matrix_i_j / self.temperature), dim=1) / (self.num_classes - 1)

            loss_i = torch.sum(-torch.log(nominator_i / (denominator_i + eps))) / self.class_counts[i]
            loss += loss_i

        return loss

    def forward(self, text, image, label):
        text = F.normalize(text, dim=1)
        image = F.normalize(image, dim=1)

        loss = self.compute_loss(text, label) + self.compute_loss(image, label)

        return loss

class CaptionLoss(nn.Module):
    def __init__(self, pad_id=0):
        super(CaptionLoss, self).__init__()
        self.pad_id = pad_id

    def forward(self, logits, text):
        # Reshape logits to (batch_size, num_classes, seq_len)
        logits = logits.permute(0, 2, 1)
        text = text.long()
        # Compute cross-entropy loss, ignoring pad_id
        caption_loss = F.cross_entropy(logits, text, ignore_index=self.pad_id)

        return caption_loss

# class CaptionLoss(torch.nn.Module):
#     def __init__(self):
#         super(CaptionLoss, self).__init__()
#
#     def forward(self, logits, text_embeds):
#         # Normalize the vectors to get cosine similarity
#         logits_norm = F.normalize(logits, p=2, dim=-1)
#         text_embeds_norm = F.normalize(text_embeds, p=2, dim=-1)
#
#         # Compute cosine similarity
#         cosine_similarity = torch.sum(logits_norm * text_embeds_norm, dim=-1)
#
#         # Compute cosine distance (1 - cosine similarity)
#         cosine_distance = 1 - cosine_similarity
#
#         # Compute the mean loss over all elements
#         loss = torch.mean(cosine_distance)
#
#         return loss

