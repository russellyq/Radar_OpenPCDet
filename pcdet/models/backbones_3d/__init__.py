from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x
from .spconv_unet import UNetV2
from .spconv_backbone_radar_thermal import VoxelBackBone8x_Radar_Thermal
from .spconv_backbone_focal import VoxelBackBone8xFocal
from .spconv_backbone_radar_thermal_crossattention import VoxelBackBone8x_Radar_Thermal_CA, VoxelBackBone8x_Radar_Thermal_Concat, VoxelBackBone8x_Radar_Thermal_Sum
from .rtdfrcnn_backbone import RTDFRCNNBackBone, RTDFRCNNBackBone_Concat, RTDFRCNNBackBone_Sum, RTDFRCNNBackBone_DeepFusionAttention

from .rtdfrcnn_backbone import RTDFRCNNBackBone_ab_study
__all__ = {
    'VoxelBackBone8x': VoxelBackBone8x,
    'UNetV2': UNetV2,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
    'VoxelResBackBone8x': VoxelResBackBone8x,
    'VoxelBackBone8x_Radar_Thermal': VoxelBackBone8x_Radar_Thermal,
    'VoxelBackBone8xFocal': VoxelBackBone8xFocal,
    'VoxelBackBone8x_Radar_Thermal_CA': VoxelBackBone8x_Radar_Thermal_CA,
    'VoxelBackBone8x_Radar_Thermal_Concat': VoxelBackBone8x_Radar_Thermal_Concat,
    'VoxelBackBone8x_Radar_Thermal_Sum': VoxelBackBone8x_Radar_Thermal_Sum,
    'RTDFRCNNBackBone': RTDFRCNNBackBone,
    'RTDFRCNNBackBone_Concat': RTDFRCNNBackBone_Concat,
    'RTDFRCNNBackBone_Sum': RTDFRCNNBackBone_Sum,
    'RTDFRCNNBackBone_DeepFusionAttention': RTDFRCNNBackBone_DeepFusionAttention,
    'RTDFRCNNBackBone_ab_study': RTDFRCNNBackBone_ab_study,
}
