# from .anchor_head_multi import AnchorHeadMulti
# from .anchor_head_single import AnchorHeadSingle, AnchorHeadSingle_bev, AnchorHeadSingle_multi, AnchorHeadSingle_multi_feature_cat
# from .anchor_head_template import AnchorHeadTemplate
# from .point_head_box import PointHeadBox
# from .point_head_simple import PointHeadSimple
# from .point_intra_part_head import PointIntraPartOffsetHead
# from .center_head import CenterHead

# __all__ = {
#     'AnchorHeadTemplate': AnchorHeadTemplate,
#     'AnchorHeadSingle': AnchorHeadSingle,
#     'AnchorHeadSingle_bev': AnchorHeadSingle_bev,
#     'AnchorHeadSingle_multi': AnchorHeadSingle_multi,
#     'AnchorHeadSingle_multi_feature_cat': AnchorHeadSingle_multi_feature_cat,
#     'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
#     'PointHeadSimple': PointHeadSimple,
#     'PointHeadBox': PointHeadBox,
#     'AnchorHeadMulti': AnchorHeadMulti,
#     'CenterHead': CenterHead
# }
from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead
from .center_head import CenterHead

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
    'AnchorHeadMulti': AnchorHeadMulti,
    'CenterHead': CenterHead
}