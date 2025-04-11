import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Prototype_Prompt_Encoder(nn.Module):
    def __init__(self, feat_dim=256, 
                        hidden_dim_dense=128, 
                        hidden_dim_sparse=128, 
                        size=64, 
                        num_landmarks = 32,
                        num_classes = 7):
                
        super(Prototype_Prompt_Encoder, self).__init__()
        self.dense_fc_1 = nn.Conv2d(feat_dim, hidden_dim_dense, 1)
        self.dense_fc_2 = nn.Conv2d(hidden_dim_dense, feat_dim, 1)
        
        self.relu = nn.ReLU()

        self.sparse_fc_1 = nn.Conv1d(size*size, hidden_dim_sparse, 1)
        self.sparse_fc_2 = nn.Conv1d(hidden_dim_sparse, num_landmarks, 1)
        
        
        pn_cls_embeddings = [nn.Embedding(num_landmarks, feat_dim) for _ in range(2)]

            
        self.pn_cls_embeddings = nn.ModuleList(pn_cls_embeddings)
        self.num_landmarks = num_landmarks
        self.num_classes = num_classes
                
    def forward(self, feat, prototypes, cls_ids):
  
        cls_prompts = prototypes.unsqueeze(-1)
        cls_prompts = torch.stack([cls_prompts for _ in range(feat.size(0))], dim=0)

        
        feat = torch.stack([feat for _ in range(cls_prompts.size(1))], dim=1)
        # compute similarity matrix 
        sim = torch.matmul(feat, cls_prompts)
        
        # compute class-activated feature
        feat =  feat + feat*sim

        feat_sparse = feat.clone()
        # compute dense embeddings
        one_hot = torch.nn.functional.one_hot(cls_ids-1,7) 
        feat = feat[one_hot ==1]
        feat = rearrange(feat,'b (h w) c -> b c h w', h=64, w=64)
        dense_embeddings = self.dense_fc_2(self.relu(self.dense_fc_1(feat)))
        
        # compute sparse embeddings
        feat_sparse = rearrange(feat_sparse,'b num_cls hw c -> (b num_cls) hw c')
        sparse_embeddings = self.sparse_fc_2(self.relu(self.sparse_fc_1(feat_sparse)))
        sparse_embeddings = rearrange(sparse_embeddings, '(b n) q c -> b n q c', n=self.num_classes)
        target_indices = cls_ids - 1
        B = sparse_embeddings.shape[0]
        predicted_sparse_target = []
        for b in range(B):
            predicted_sparse_target.append(sparse_embeddings[b, target_indices[b], ...]) 
        predicted_sparse_target = torch.stack(predicted_sparse_target, dim=0)
        pos_embed = self.pn_cls_embeddings[1].weight.unsqueeze(0).unsqueeze(0) * one_hot.unsqueeze(-1).unsqueeze(-1)
        neg_embed = self.pn_cls_embeddings[0].weight.unsqueeze(0).unsqueeze(0) * (1-one_hot).unsqueeze(-1).unsqueeze(-1)
        

        sparse_embeddings = sparse_embeddings + pos_embed.detach() + neg_embed.detach()
            
        sparse_embeddings = rearrange(sparse_embeddings,'b n q c -> b (n q) c')
        
        return dense_embeddings, sparse_embeddings, predicted_sparse_target

class Learnable_Prototypes(nn.Module):
    def __init__(self, num_classes=7 , feat_dim=256):
        super(Learnable_Prototypes, self).__init__()
        self.class_embeddings = nn.Embedding(num_classes, feat_dim)
        
    def forward(self):
        return self.class_embeddings.weight
    
class PrototypeAwareMultiModalEncoder(nn.Module):
    def __init__(self, feat_dim=256, 
                 hidden_dim=128,
                 feat_size=64, 
                 num_landmarks=32,
                 num_classes=7,
                 num_heads=8,
                 dropout=0.1,
                 use_gating=True):
        super(PrototypeAwareMultiModalEncoder, self).__init__()
        
        self.multi_scale_processor = MultiScaleFeatureProcessor(
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            scales=[1, 2, 4]
        )
        
        self.cross_modal_attention = CrossModalAttention(
            feat_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.dense_gate = AdaptiveFeatureGate(feat_dim, hidden_dim) if use_gating else nn.Identity()
        
        self.dense_branch = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Conv2d(feat_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim),
            nn.Conv2d(hidden_dim, feat_dim, 1)
        )
        
        self.sparse_processor = SparseLandmarkProcessor(
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            num_landmarks=num_landmarks,
            dropout=dropout
        )
        
        self.prototype_consistency = PrototypeConsistencyModule(
            feat_dim=feat_dim,
            num_classes=num_classes
        )
        
        self.pos_embeddings = nn.ModuleList([
            nn.Embedding(num_landmarks, feat_dim) for _ in range(2)
        ])
        
        self.feat_dim = feat_dim
        self.feat_size = feat_size
        self.num_landmarks = num_landmarks
        self.num_classes = num_classes
        
    def forward(self, features, prototypes, cls_ids, return_attn=False):
        B, HW, C = features.shape
        H = W = self.feat_size
        
        features_ms = rearrange(features, 'b (h w) c -> b c h w', h=H, w=W)
        features_ms = self.multi_scale_processor(features_ms)
        features = rearrange(features_ms, 'b c h w -> b (h w) c')
        
        one_hot = F.one_hot(cls_ids-1, self.num_classes).float()  # (B, num_classes)
        
        enhanced_features, attn_weights = self.cross_modal_attention(
            features, prototypes, cls_ids)
        
        dense_features = enhanced_features[one_hot.unsqueeze(1).expand(-1, HW, -1) == 1]
        dense_features = rearrange(dense_features, 'b (h w) c -> b c h w', h=H, w=W)
        
        dense_features = self.dense_gate(dense_features)
        dense_embeddings = self.dense_branch(dense_features)
        sparse_embeddings, predicted_landmarks = self.sparse_processor(
            enhanced_features, cls_ids, self.num_classes)
        
        prototype_consistency_loss = self.prototype_consistency(
            enhanced_features, prototypes, cls_ids)
        
        pos_embed = self.pos_embeddings[1].weight.unsqueeze(0).unsqueeze(0) * one_hot.unsqueeze(-1).unsqueeze(-1)
        neg_embed = self.pos_embeddings[0].weight.unsqueeze(0).unsqueeze(0) * (1-one_hot).unsqueeze(-1).unsqueeze(-1)
        
        sparse_embeddings = sparse_embeddings + pos_embed.detach() + neg_embed.detach()
        
        sparse_embeddings = rearrange(sparse_embeddings, 'b n q c -> b (n q) c')
        
        if return_attn:
            return dense_embeddings, sparse_embeddings, predicted_landmarks, attn_weights, prototype_consistency_loss
        else:
            return dense_embeddings, sparse_embeddings, predicted_landmarks


class MultiScaleFeatureProcessor(nn.Module):
    def __init__(self, feat_dim, hidden_dim, scales=[1, 2, 4]):
        super(MultiScaleFeatureProcessor, self).__init__()
        self.scales = scales
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool2d(s, stride=s) if s > 1 else nn.Identity(),
                nn.Conv2d(feat_dim, hidden_dim, 1),
                nn.GELU()
            ) for s in scales
        ])
        self.fusion = nn.Conv2d(hidden_dim * len(scales), feat_dim, 1)
        self.norm = nn.LayerNorm([feat_dim, 1, 1])
        
    def forward(self, x):
        multi_scale_feats = []
        
        for i, scale in enumerate(self.scales):
            feat_scale = self.proj_layers[i](x)
            
            # Upsample if need
            if scale > 1:
                feat_scale = F.interpolate(
                    feat_scale, size=x.shape[2:], 
                    mode='bilinear', align_corners=False
                )
                
            multi_scale_feats.append(feat_scale)
            
        concat_feats = torch.cat(multi_scale_feats, dim=1)
        output = self.fusion(concat_feats)
        
        output = output + x
        
        output = self.norm(output)
        
        return output

class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias
    
class CrossModalAttention(nn.Module):
    def __init__(self, feat_dim, num_heads=8, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        assert feat_dim % num_heads == 0, "feat_dim must be divisible by num_heads"
        
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        
        # Projection layers
        self.query_proj = nn.Linear(feat_dim, feat_dim)
        self.key_proj = nn.Linear(feat_dim, feat_dim)
        self.value_proj = nn.Linear(feat_dim, feat_dim)
        self.output_proj = nn.Linear(feat_dim, feat_dim)
        
        # Attention dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # self.norm1 = nn.LayerNorm(feat_dim)
        # self.norm2 = nn.LayerNorm(feat_dim)
        
        self.norm1 = DyT(feat_dim)
        self.norm2 = DyT(feat_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 4, feat_dim)
        )
        
    def forward(self, features, prototypes, cls_ids):
        batch_size, seq_len, _ = features.shape
        
        features = self.norm1(features)
        
        proto_batch = prototypes.expand(batch_size, -1, -1)  # (B, num_classes, C)
        
        q = self.query_proj(proto_batch)  # Prototypes as queries
        k = self.key_proj(features)       # Features as keys
        v = self.value_proj(features)     # Features as values
        
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, num_classes, D/H)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, seq_len, D/H)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, seq_len, D/H)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, num_classes, seq_len)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)  # (B, H, num_classes, D/H)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.feat_dim)  # (B, num_classes, D)
        attn_output = self.output_proj(attn_output)
        
        class_attn = attn_weights.mean(dim=1)
        
        expanded_features = features.unsqueeze(1).expand(-1, prototypes.size(0), -1, -1)  # (B, num_classes, seq_len, D)
        
        modulated_features = expanded_features * class_attn.unsqueeze(-1)  # (B, num_classes, seq_len, D)
        
        enhanced_features = expanded_features + modulated_features  # (B, num_classes, seq_len, D)
        
        return enhanced_features, class_attn


class AdaptiveFeatureGate(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(AdaptiveFeatureGate, self).__init__()
        self.gate_network = nn.Sequential(
            nn.Conv2d(feat_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, feat_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        gates = self.gate_network(x)
        return x * gates


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels)
        )
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out = out + residual
        out = self.activation(out)
        return out

class SparseLandmarkProcessor(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_landmarks, dropout=0.1):
        super(SparseLandmarkProcessor, self).__init__()
        
        self.feature_transform = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.landmark_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_landmarks)
        )
        
        self.output_proj = nn.Linear(hidden_dim, feat_dim)
        
        self.num_landmarks = num_landmarks
        
    def forward(self, features, cls_ids, num_classes):
        B, num_cls, seq_len, C = features.shape
        
        flat_features = rearrange(features, 'b n s c -> (b n) s c')
        
        transformed = self.feature_transform(flat_features)  # (B*num_classes, seq_len, hidden_dim)
        
        landmark_weights = self.landmark_predictor(transformed)  # (B*num_classes, seq_len, num_landmarks)
        landmark_weights = F.softmax(landmark_weights, dim=1)  # Normalize across sequence dimension
        landmark_features = torch.bmm(
            landmark_weights.transpose(1, 2),  # (B*num_classes, num_landmarks, seq_len)
            transformed  # (B*num_classes, seq_len, hidden_dim)
        )  # (B*num_classes, num_landmarks, hidden_dim)
        
        landmark_features = self.output_proj(landmark_features)  # (B*num_classes, num_landmarks, feat_dim)
        sparse_embeddings = rearrange(
            landmark_features, '(b n) q c -> b n q c', b=B, n=num_cls
        )  # (B, num_classes, num_landmarks, feat_dim)
        
        target_indices = cls_ids - 1  # Convert to 0-indexed
        predicted_landmarks = []
        for b in range(B):
            predicted_landmarks.append(sparse_embeddings[b, target_indices[b]])  # (num_landmarks, feat_dim)
        predicted_landmarks = torch.stack(predicted_landmarks, dim=0)  # (B, num_landmarks, feat_dim)
        
        return sparse_embeddings, predicted_landmarks

class PrototypeConsistencyModule(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super(PrototypeConsistencyModule, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        self.projection = nn.Linear(feat_dim, feat_dim)
        self.num_classes = num_classes
        
    def forward(self, features, prototypes, cls_ids):
        B, num_cls, seq_len, C = features.shape
        
        one_hot = F.one_hot(cls_ids-1, self.num_classes).float()  # (B, num_classes)
        class_mask = one_hot.unsqueeze(2).expand(-1, -1, seq_len)  # (B, num_classes, seq_len)
        
        target_features = []
        for b in range(B):
            target_idx = cls_ids[b] - 1
            target_feat = features[b, target_idx].mean(dim=0)
            target_features.append(target_feat)
        target_features = torch.stack(target_features, dim=0)  # (B, C)
        
        projected_features = self.projection(target_features)  # (B, C)
        projected_prototypes = self.projection(prototypes)  # (num_classes, C)
        
        similarity = torch.matmul(
            projected_features, projected_prototypes.t()
        ) / self.temperature  # (B, num_classes)
        
        labels = cls_ids - 1  # Convert to 0-indexed
        loss = F.cross_entropy(similarity, labels)
        
        return loss