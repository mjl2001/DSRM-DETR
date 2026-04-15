# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
from mmengine.model import ModuleList
from torch import Tensor, nn

from .detr_layers import (DetrTransformerDecoder, DetrTransformerDecoderLayer,
                          DetrTransformerEncoder, DetrTransformerEncoderLayer)
from .utils import inverse_sigmoid


class DynamicMaskDeformableAttention(MultiScaleDeformableAttention):
    """Dynamic Mask Deformable Attention.
    
    根据预测框面积动态调整采样点数量：
    - 大目标：只使用前4个采样点
    - 中目标：只使用前8个采样点
    - 小目标：使用全部采样点
    
    Args:
        small_thresh (float): 小目标面积阈值。默认为0.02。
        large_thresh (float): 大目标面积阈值。默认为0.1。
    """
    
    def __init__(self, 
                 *args,
                 small_thresh: float = 0.02,
                 large_thresh: float = 0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.small_thresh = small_thresh
        self.large_thresh = large_thresh
    
    def apply_dynamic_mask(self, 
                          attn_weights: Tensor, 
                          bbox_areas: Tensor) -> Tensor:
        """应用动态mask到attention weights。
        
        Args:
            attn_weights (Tensor): 注意力权重，形状为 (bs, num_query, heads, levels, points)
            bbox_areas (Tensor): 预测框面积，形状为 (bs, num_query)
            
        Returns:
            Tensor: Masked后的注意力权重，形状为 (bs, num_query, heads, levels, points)
        """
        bs, nq, heads, levels, points = attn_weights.shape
        
        # 检查bbox_areas的值范围
        if bbox_areas.min() < 0 or bbox_areas.max() > 1:
            # 如果超出范围，进行clip
            bbox_areas = bbox_areas.clamp(min=0.0, max=1.0)
        
        # 向量化实现：生成mask
        # bbox_areas: (bs, num_query) -> (bs, num_query, 1, 1, 1)
        areas = bbox_areas.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # 创建mask (bs, num_query, heads, levels, points)
        mask = torch.ones_like(attn_weights)
        
        # 小目标mask: area <= small_thresh, 保留全部点
        small_mask = areas <= self.small_thresh
        
        # 中目标mask: small_thresh < area <= large_thresh, 只保留前8个点
        mid_mask = (areas > self.small_thresh) & (areas <= self.large_thresh)
        if points > 8:
            mid_point_mask = torch.cat([
                torch.ones_like(mask[..., :8]),
                torch.zeros_like(mask[..., 8:])
            ], dim=-1)
            mask = torch.where(mid_mask.expand_as(mask), mid_point_mask, mask)
        # 如果points <= 8，中目标保留全部点（不需要额外处理）
        
        # 大目标mask: area > large_thresh, 只保留前4个点
        large_mask = areas > self.large_thresh
        if points > 4:
            large_point_mask = torch.cat([
                torch.ones_like(mask[..., :4]),
                torch.zeros_like(mask[..., 4:])
            ], dim=-1)
            mask = torch.where(large_mask.expand_as(mask), large_point_mask, mask)
        # 如果points <= 4，大目标保留全部点（不需要额外处理）
        
        # 应用mask
        attn_weights = attn_weights * mask
        
        # 重新归一化
        attn_weights = attn_weights / (attn_weights.sum(-1, keepdim=True) + 1e-6)
        
        return attn_weights
    
    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_padding_mask: Tensor = None,
                reference_points: Tensor = None,
                spatial_shapes: Tensor = None,
                level_start_index: Tensor = None,
                bbox_areas: Tensor = None,
                **kwargs) -> Tensor:
        """Forward function with dynamic mask support.
        
        Args:
            query (Tensor): Query embeddings, shape (bs, num_query, embed_dims).
            key (Tensor): Key embeddings (not used, for compatibility).
            value (Tensor): Value embeddings, shape (bs, num_value, embed_dims).
            query_pos (Tensor): Positional encoding for query.
            key_padding_mask (Tensor): Key padding mask.
            reference_points (Tensor): Reference points, shape (bs, num_query, num_levels, 2).
            spatial_shapes (Tensor): Spatial shapes of features.
            level_start_index (Tensor): Start index of each level.
            bbox_areas (Tensor): Predicted bbox areas, shape (bs, num_query).
            
        Returns:
            Tensor: Output features, shape (bs, num_query, embed_dims).
        """
        if value is None:
            value = query
        
        if query_pos is not None:
            query = query + query_pos
        
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        
        # Value projection
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        
        # Sampling offsets
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        
        # Attention weights
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, -1)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points)
        
        # ===== 应用动态mask =====
        if bbox_areas is not None:
            attention_weights = self.apply_dynamic_mask(attention_weights, bbox_areas)
        
        # Deformable attention
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + \
                sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + \
                sampling_offsets / self.num_points * \
                reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be 2 or 4, '
                f'but got {reference_points.shape[-1]}')
        
        # 使用PyTorch版本的deformable attention
        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights)
        
        # Output projection
        output = self.output_proj(output)
        
        return output


class TPRCM(nn.Module):
    """Target-aware Reference Point Confidence Modulation Module.
    
    该模块用于在可变形交叉注意力计算之前对参考点的有效性进行评估与调制，
    从而缓解复杂背景下无效采样点对小目标特征建模带来的干扰。
    
    Args:
        embed_dims (int): 特征维度。默认为256。
        hidden_dims (int): MLP隐藏层维度。默认为64。
        num_levels (int): 多尺度特征层数。默认为4。
    """
    
    def __init__(self, 
                 embed_dims: int = 256, 
                 hidden_dims: int = 64,
                 num_levels: int = 4):
        super().__init__()
        self.embed_dims = embed_dims
        self.hidden_dims = hidden_dims
        self.num_levels = num_levels
        
        # 轻量级MLP用于计算可信度权重
        # 输入: 局部特征 + 查询特征 (embed_dims * 2)
        # 输出: 可信度权重 (1)
        self.confidence_mlp = nn.Sequential(
            nn.Linear(embed_dims * 2, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.confidence_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def sample_local_features(self,
                              value: Tensor,
                              reference_points: Tensor,
                              spatial_shapes: Tensor,
                              level_start_index: Tensor) -> Tensor:
        """在多尺度特征图上采样参考点处的局部特征。
        
        Args:
            value (Tensor): 编码器输出的多尺度特征，形状为 (bs, num_value, embed_dims)
            reference_points (Tensor): 参考点坐标，形状为 (bs, num_queries, num_levels, 2)
                或 (bs, num_queries, num_levels, 4)，只使用前两维 (cx, cy)
            spatial_shapes (Tensor): 各层特征图的空间尺寸，形状为 (num_levels, 2)
            level_start_index (Tensor): 各层特征的起始索引，形状为 (num_levels,)
            
        Returns:
            Tensor: 采样得到的局部特征，形状为 (bs, num_queries, embed_dims)
        """
        bs, num_queries, num_levels, ref_dim = reference_points.shape
        
        # 只使用参考点的前两维 (cx, cy)
        if ref_dim == 4:
            reference_points = reference_points[..., :2]
        
        sampled_features_list = []
        
        for lvl in range(num_levels):
            # 将 Tensor 转换为 int 用于 view 操作
            H = int(spatial_shapes[lvl, 0].item())
            W = int(spatial_shapes[lvl, 1].item())
            start_idx = int(level_start_index[lvl].item())
            if lvl < num_levels - 1:
                end_idx = int(level_start_index[lvl + 1].item())
            else:
                end_idx = value.shape[1]
            
            # 获取当前层的特征 (bs, H*W, embed_dims)
            lvl_value = value[:, start_idx:end_idx, :]
            # 重塑为 (bs, embed_dims, H, W)
            lvl_value = lvl_value.view(bs, H, W, -1).permute(0, 3, 1, 2).contiguous()
            
            # 获取当前层的参考点 (bs, num_queries, 2)
            lvl_ref_points = reference_points[:, :, lvl, :]
            
            # 将参考点坐标从 [0, 1] 转换为 [-1, 1] 用于 grid_sample
            # 注意: grid_sample 期望的坐标顺序是 (x, y)，参考点已经是 (x, y) 格式
            grid = lvl_ref_points * 2 - 1  # (bs, num_queries, 2)
            grid = grid.unsqueeze(2)  # (bs, num_queries, 1, 2)
            
            # 双线性插值采样 (bs, embed_dims, num_queries, 1)
            sampled = F.grid_sample(
                lvl_value, 
                grid, 
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=False
            )
            # (bs, embed_dims, num_queries, 1) -> (bs, num_queries, embed_dims)
            sampled = sampled.squeeze(-1).permute(0, 2, 1)
            sampled_features_list.append(sampled)
        
        # 对多尺度特征取平均 (bs, num_queries, embed_dims)
        sampled_features = torch.stack(sampled_features_list, dim=0).mean(dim=0)
        
        return sampled_features
    
    def forward(self,
                query: Tensor,
                value: Tensor,
                reference_points: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor) -> Tensor:
        """计算参考点的可信度权重。
        
        Args:
            query (Tensor): 对象查询特征，形状为 (bs, num_queries, embed_dims)
            value (Tensor): 编码器输出的多尺度特征，形状为 (bs, num_value, embed_dims)
            reference_points (Tensor): 参考点坐标，形状为 (bs, num_queries, num_levels, 2)
                或 (bs, num_queries, num_levels, 4)，只使用前两维 (cx, cy)
            spatial_shapes (Tensor): 各层特征图的空间尺寸，形状为 (num_levels, 2)
            level_start_index (Tensor): 各层特征的起始索引，形状为 (num_levels,)
            
        Returns:
            Tensor: 可信度权重，形状为 (bs, num_queries, 1)
        """
        # 采样参考点处的局部特征 (bs, num_queries, embed_dims)
        local_features = self.sample_local_features(
            value, reference_points, spatial_shapes, level_start_index
        )
        
        # 将局部特征与查询特征在通道维度上融合 (bs, num_queries, embed_dims * 2)
        fused_features = torch.cat([local_features, query], dim=-1)
        
        # 通过MLP计算可信度权重 (bs, num_queries, 1)
        confidence_weights = self.confidence_mlp(fused_features)
        
        return confidence_weights

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None


class DeformableDetrTransformerEncoder(DetrTransformerEncoder):
    """Transformer encoder of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])

        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])

        self.embed_dims = self.layers[0].embed_dims

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, spatial_shapes: Tensor,
                level_start_index: Tensor, valid_ratios: Tensor,
                **kwargs) -> Tensor:
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, has shape
                (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            Tensor: Output queries of Transformer encoder, which is also
            called 'encoder output embeddings' or 'memory', has shape
            (bs, num_queries, dim)
        """
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        for layer in self.layers:
            query = layer(
                query=query,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points,
                **kwargs)
        return query

    @staticmethod
    def get_encoder_reference_points(
            spatial_shapes: Tensor, valid_ratios: Tensor,
            device: Union[torch.device, str]) -> Tensor:
        """Get the reference points used in encoder.

        Args:
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            device (obj:`device` or str): The device acquired by the
                `reference_points`.

        Returns:
            Tensor: Reference points used in decoder, has shape (bs, length,
            num_levels, 2).
        """

        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        # [bs, sum(hw), num_level, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points


class DeformableDetrTransformerDecoder(DetrTransformerDecoder):
    """Transformer Decoder of Deformable DETR with TPRCM support.
    
    Args:
        use_tprcm (bool): 是否使用TPRCM模块。默认为False。
        tprcm_cfg (dict): TPRCM模块的配置。默认为None。
        tprcm_start_layer (int): TPRCM开始启用的层索引（从0开始）。默认为1（即第二层）。
        use_dynamic_mask (bool): 是否使用动态mask。默认为False。
    """

    def __init__(self,
                 *args,
                 use_tprcm: bool = False,
                 tprcm_cfg: Optional[dict] = None,
                 tprcm_start_layer: int = 1,
                 use_dynamic_mask: bool = False,
                 **kwargs):
        self.use_tprcm = use_tprcm
        self.tprcm_cfg = tprcm_cfg or {}
        self.tprcm_start_layer = tprcm_start_layer
        self.use_dynamic_mask = use_dynamic_mask
        
        # 警告：动态mask需要with_box_refine=True
        if self.use_dynamic_mask:
            import warnings
            warnings.warn(
                "Dynamic mask is enabled. Make sure 'with_box_refine=True' "
                "in your model config, otherwise bbox_areas will be None and "
                "dynamic mask will not work.",
                UserWarning
            )
        
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize decoder layers and TPRCM module."""
        # 如果使用动态mask，需要在layer_cfg中添加标记
        import copy
        layer_cfg = copy.deepcopy(self.layer_cfg)  # 使用深拷贝避免修改原配置
        if self.use_dynamic_mask:
            if 'cross_attn_cfg' not in layer_cfg:
                layer_cfg['cross_attn_cfg'] = {}
            layer_cfg['cross_attn_cfg']['use_dynamic_mask'] = True
        
        self.layers = ModuleList([
            DeformableDetrTransformerDecoderLayer(**layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        
        # 初始化TPRCM模块（所有解码器层共享参数）
        if self.use_tprcm:
            tprcm_cfg = dict(embed_dims=self.embed_dims)
            tprcm_cfg.update(self.tprcm_cfg)
            self.tprcm = TPRCM(**tprcm_cfg)

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                value: Tensor,
                key_padding_mask: Tensor,
                reference_points: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                reg_branches: Optional[nn.Module] = None,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input queries, has shape (bs, num_queries,
                dim).
            query_pos (Tensor): The input positional query, has shape
                (bs, num_queries, dim). It will be added to `query` before
                forward function.
            value (Tensor): The input values, has shape (bs, num_value, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `cross_attn`
                input. ByteTensor, has shape (bs, num_value).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h) when `as_two_stage` is `True`, otherwise has
                shape (bs, num_queries, 2) with the last dimension arranged
                as (cx, cy).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`, optional): Used for refining
                the regression results. Only would be passed when
                `with_box_refine` is `True`, otherwise would be `None`.

        Returns:
            tuple[Tensor]: Outputs of Deformable Transformer Decoder.

            - output (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * \
                    valid_ratios[:, None]
            
            # 计算TPRCM可信度权重（从第二层开始启用）
            confidence_weights = None
            if self.use_tprcm and layer_id >= self.tprcm_start_layer:
                confidence_weights = self.tprcm(
                    query=output,
                    value=value,
                    reference_points=reference_points_input,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index
                )
            
            # 计算bbox_areas用于动态mask
            bbox_areas = None
            if self.use_dynamic_mask and reg_branches is not None:
                # 通过回归分支预测bbox
                tmp_bbox = reg_branches[layer_id](output)
                if reference_points.shape[-1] == 4:
                    # 如果reference_points是4维的，直接使用预测的bbox
                    bbox_pred = tmp_bbox + inverse_sigmoid(reference_points)
                    bbox_pred = bbox_pred.sigmoid()
                else:
                    # 如果reference_points是2维的，tmp_bbox应该是4维(cx, cy, w, h)
                    # 创建副本避免修改原tensor
                    bbox_pred = tmp_bbox.clone()
                    bbox_pred[..., :2] = tmp_bbox[..., :2] + inverse_sigmoid(reference_points)
                    bbox_pred = bbox_pred.sigmoid()
                
                # 计算面积: w * h
                # 确保w和h都是正值，并限制在合理范围内
                w = bbox_pred[..., 2].clamp(min=1e-6, max=1.0)
                h = bbox_pred[..., 3].clamp(min=1e-6, max=1.0)
                bbox_areas = w * h
            
            output = layer(
                output,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                confidence_weights=confidence_weights,
                bbox_areas=bbox_areas,
                **kwargs)

            if reg_branches is not None:
                tmp_reg_preds = reg_branches[layer_id](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp_reg_preds + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp_reg_preds
                    new_reference_points[..., :2] = tmp_reg_preds[
                        ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


class DeformableDetrTransformerEncoderLayer(DetrTransformerEncoderLayer):
    """Encoder layer of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize self_attn, ffn, and norms."""
        self.self_attn = MultiScaleDeformableAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)


class DeformableDetrTransformerDecoderLayer(DetrTransformerDecoderLayer):
    """Decoder layer of Deformable DETR with TPRCM support."""

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        
        # 检查是否使用动态mask
        cross_attn_cfg = self.cross_attn_cfg.copy()
        use_dynamic_mask = cross_attn_cfg.pop('use_dynamic_mask', False)
        
        if use_dynamic_mask:
            # 使用动态mask版本的deformable attention
            self.cross_attn = DynamicMaskDeformableAttention(**cross_attn_cfg)
        else:
            # 使用标准的deformable attention
            self.cross_attn = MultiScaleDeformableAttention(**cross_attn_cfg)
        
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                reference_points: Tensor = None,
                spatial_shapes: Tensor = None,
                level_start_index: Tensor = None,
                confidence_weights: Tensor = None,
                bbox_areas: Tensor = None,
                **kwargs) -> Tensor:
        """Forward function of decoder layer with TPRCM and dynamic mask support.
        
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key. Defaults to None.
            value (Tensor, optional): The input value. Defaults to None.
            query_pos (Tensor, optional): The positional encoding for query.
            key_pos (Tensor, optional): The positional encoding for key.
            self_attn_mask (Tensor, optional): Self attention mask.
            cross_attn_mask (Tensor, optional): Cross attention mask.
            key_padding_mask (Tensor, optional): Key padding mask.
            reference_points (Tensor, optional): Reference points for 
                deformable attention.
            spatial_shapes (Tensor, optional): Spatial shapes of features.
            level_start_index (Tensor, optional): Start index of each level.
            confidence_weights (Tensor, optional): TPRCM confidence weights,
                has shape (bs, num_queries, 1). If provided, will be used to
                modulate the cross attention output.
            bbox_areas (Tensor, optional): Predicted bbox areas for dynamic mask,
                has shape (bs, num_queries). If provided, will be used to
                dynamically mask attention weights.
            **kwargs: Additional arguments.
            
        Returns:
            Tensor: Forwarded results, has shape (bs, num_queries, dim).
        """
        # Self attention
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[0](query)
        
        # Cross attention (Deformable Attention)
        # 根据attention类型决定是否传递bbox_areas
        if isinstance(self.cross_attn, DynamicMaskDeformableAttention):
            # 动态mask版本，传递bbox_areas
            cross_attn_output = self.cross_attn(
                query=query,
                key=value,
                value=value,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                bbox_areas=bbox_areas,
                **kwargs)
        else:
            # 标准版本，不传递bbox_areas
            cross_attn_output = self.cross_attn(
                query=query,
                key=value,
                value=value,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                **kwargs)
        
        # 应用TPRCM可信度权重调制
        if confidence_weights is not None:
            # confidence_weights: (bs, num_queries, 1)
            # cross_attn_output: (bs, num_queries, embed_dims)
            # 使用可信度权重对交叉注意力输出进行调制
            cross_attn_output = cross_attn_output * confidence_weights
        
        query = query + cross_attn_output
        query = self.norms[1](query)
        
        # FFN
        query = self.ffn(query)
        query = self.norms[2](query)

        return query
