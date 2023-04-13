from typing import Dict, List, Optional, Tuple
from typing import Tuple, List, Dict, Optional, Union
from collections import OrderedDict
import torch
import warnings
from torch import nn, Tensor
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision
from torchvision.ops import boxes as box_ops

from torchvision.models.detection import _utils as det_utils
# from pcdet.models.backbones_3d.thermal_resnet import resnet18
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, mobilenet_backbone, _validate_trainable_layers
from torchvision.ops import MultiScaleRoIAlign
# Import AnchorGenerator to keep compatibility.
from torchvision.models.detection.anchor_utils import AnchorGenerator  # noqa: 401
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
# from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.models.detection.rpn import RPNHead

from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor


@torch.jit.unused
def _onnx_get_num_anchors_and_pre_nms_top_n(ob, orig_pre_nms_top_n):
    # type: (Tensor, int) -> Tuple[int, int]
    from torch.onnx import operators
    num_anchors = operators.shape_as_tensor(ob)[1].unsqueeze(0)
    pre_nms_top_n = torch.min(torch.cat(
        (torch.tensor([orig_pre_nms_top_n], dtype=num_anchors.dtype),
         num_anchors), 0))

    return num_anchors, pre_nms_top_n




def _default_anchorgen():
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)

# # class RPNHead(nn.Module):
# #     """
# #     Adds a simple RPN Head with classification and regression heads
# #     Args:
# #         in_channels (int): number of channels of the input feature
# #         num_anchors (int): number of anchors to be predicted
# #         conv_depth (int, optional): number of convolutions
# #     """

# #     _version = 2

# #     def __init__(self, in_channels: int, num_anchors: int, conv_depth=1) -> None:
# #         super().__init__()
# #         convs = []
# #         for _ in range(conv_depth):
# #             convs.append(Conv2dNormActivation(in_channels, in_channels, kernel_size=3, norm_layer=None))
# #         self.conv = nn.Sequential(*convs)
# #         self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
# #         self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

# #         for layer in self.modules():
# #             if isinstance(layer, nn.Conv2d):
# #                 torch.nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
# #                 if layer.bias is not None:
# #                     torch.nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]

# #     def _load_from_state_dict(
# #         self,
# #         state_dict,
# #         prefix,
# #         local_metadata,
# #         strict,
# #         missing_keys,
# #         unexpected_keys,
# #         error_msgs,
# #     ):
# #         version = local_metadata.get("version", None)

# #         if version is None or version < 2:
# #             for type in ["weight", "bias"]:
# #                 old_key = f"{prefix}conv.{type}"
# #                 new_key = f"{prefix}conv.0.0.{type}"
# #                 if old_key in state_dict:
# #                     state_dict[new_key] = state_dict.pop(old_key)

# #         super()._load_from_state_dict(
# #             state_dict,
# #             prefix,
# #             local_metadata,
# #             strict,
# #             missing_keys,
# #             unexpected_keys,
# #             error_msgs,
# #         )

# #     def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
# #         logits = []
# #         bbox_reg = []
# #         for feature in x:
# #             t = self.conv(feature)
# #             logits.append(self.cls_logits(t))
# #             bbox_reg.append(self.bbox_pred(t))
# #         return logits, bbox_reg

# class RPNHead(nn.Module):
#     """
#     Adds a simple RPN Head with classification and regression heads
#     Args:
#         in_channels (int): number of channels of the input feature
#         num_anchors (int): number of anchors to be predicted
#     """

#     def __init__(self, in_channels, num_anchors):
#         super(RPNHead, self).__init__()
#         self.conv = nn.Conv2d(
#             in_channels, in_channels, kernel_size=3, stride=1, padding=1
#         )
#         self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
#         self.bbox_pred = nn.Conv2d(
#             in_channels, num_anchors * 4, kernel_size=1, stride=1
#         )

#         for layer in self.children():
#             torch.nn.init.normal_(layer.weight, std=0.01)
#             torch.nn.init.constant_(layer.bias, 0)

#     def forward(self, x):
#         # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
#         logits = []
#         bbox_reg = []
#         for feature in x:
#             t = F.relu(self.conv(feature))
#             logits.append(self.cls_logits(t))
#             bbox_reg.append(self.bbox_pred(t))
#         return logits, bbox_reg



def permute_and_flatten(layer, N, A, C, H, W):
    # type: (Tensor, int, int, int, int, int) -> Tensor
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    # type: (List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(
        box_cls, box_regression
    ):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 4, H, W
        )
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


class RegionProposalNetwork(torch.nn.Module):
    """
    Implements Region Proposal Network (RPN).
    Args:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): module that computes the objectness and regression deltas
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        pre_nms_top_n (Dict[str, int]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_top_n (Dict[str, int]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
    """

    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sampler": det_utils.BalancedPositiveNegativeSampler,
    }

    # def __init__(
    #     self,
    #     anchor_generator: AnchorGenerator,
    #     head: nn.Module,
    #     # Faster-RCNN Training
    #     fg_iou_thresh: float,
    #     bg_iou_thresh: float,
    #     batch_size_per_image: int,
    #     positive_fraction: float,
    #     # Faster-RCNN Inference
    #     pre_nms_top_n: Dict[str, int],
    #     post_nms_top_n: Dict[str, int],
    #     nms_thresh: float,
    #     score_thresh: float = 0.0,
    # ) -> None:
    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        head: nn.Module,
        # Faster-RCNN Training
        fg_iou_thresh: float = 0.7,
        bg_iou_thresh: float = 0.3,
        batch_size_per_image: int = 256,
        positive_fraction: float = 0.5,
        # Faster-RCNN Inference
        pre_nms_top_n: Dict[str, int] = dict(training=2000, testing=1000),
        post_nms_top_n: Dict[str, int] = dict(training=2000, testing=1000),
        nms_thresh: float = 0.7,
        score_thresh: float = 0.0,
    ) -> None:
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # used during training
        self.box_similarity = box_ops.box_iou

        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )
        # used during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']


    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        # type: (Tensor, List[int]) -> Tensor
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            if torchvision._is_tracing():
                num_anchors, pre_nms_top_n = _onnx_get_num_anchors_and_pre_nms_top_n(ob, self.pre_nms_top_n())
            else:
                num_anchors = ob.shape[1]
                pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        # type: (Tensor, Tensor, List[Tuple[int, int]], List[int]) -> Tuple[List[Tensor], List[Tensor]]
        num_images = proposals.shape[0]
        device = proposals.device
        # do not backprop throught objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        levels = [
            torch.full((n,), idx, dtype=torch.int64, device=device)
            for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]

        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # remove low scoring boxes
            # use >= for Backwards compatibility
            keep = torch.where(scores >= self.score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[:self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def forward(self,
                images,       # type: ImageList
                features,     # type: Dict[str, Tensor]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Tensor], Dict[str, Tensor]]
        """
        Args:
            images (ImageList): images for which we want to compute the predictions
            features (OrderedDict[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.
        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = \
            concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)
        return boxes, scores
        



class RCNN(nn.Module):
    """
    Main class for Generalized R-CNN.
    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self):
        super(RCNN, self).__init__()
        num_classes=2
        pretrained=False
        pretrained_backbone=True
        trainable_backbone_layers = _validate_trainable_layers(pretrained or pretrained_backbone, None, 5, 3)
        backbone = resnet_fpn_backbone('resnet18', pretrained_backbone, trainable_layers=trainable_backbone_layers)
        
        rpn_anchor_generator = _default_anchorgen()
        out_channels = backbone.out_channels
        rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        # rpn_head = rpn_head.to(device)
        rpn = RegionProposalNetwork(
                rpn_anchor_generator,
                rpn_head,
                fg_iou_thresh=0.7,
                bg_iou_thresh=0.3,
                batch_size_per_image=256,
                positive_fraction=0.5,
                pre_nms_top_n= dict(training=2000, testing=1000),
                post_nms_top_n= dict(training=2000, testing=1000),
                nms_thresh=0.7, score_thresh=0.0
            )
        # rpn = rpn.to(device)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        min_size=800
        max_size=1333
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        
        box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)
        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)
        representation_size = 1024
        box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)
        box_fg_iou_thresh=0.5
        box_bg_iou_thresh=0.5
        box_batch_size_per_image=512
        box_positive_fraction=0.25
        bbox_reg_weights=None
        box_score_thresh=0.05
        box_nms_thresh=0.5
        box_detections_per_img=100
        roi_heads = RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            proposals: (list[Tensor])
            proposal_losses: (list[Tensor])
            feature_lists[0][1]: (B, C, H, W)
        
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        # if self.training and targets is None:
        #     raise ValueError("In training mode, targets should be passed")
        # if self.training:
        #     assert targets is not None
        #     for target in targets:
        #         boxes = target["boxes"]
        #         if isinstance(boxes, torch.Tensor):
        #             if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
        #                 raise ValueError("Expected target boxes to be a tensor"
        #                                  "of shape [N, 4], got {:}.".format(
        #                                      boxes.shape))
        #         else:
        #             raise ValueError("Expected target boxes to be of type "
        #                              "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        # print('original_image_sizes', original_image_sizes)
        images, targets = self.transform(images, targets)

        # # Check for degenerate boxes
        # # TODO: Move this to a function
        # if targets is not None:
        #     for target_idx, target in enumerate(targets):
        #         boxes = target["boxes"]
        #         degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
        #         if degenerate_boxes.any():
        #             # print the first degenerate box
        #             bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
        #             degen_bb: List[float] = boxes[bb_idx].tolist()
        #             raise ValueError("All bounding boxes should have positive height and width."
        #                              " Found invalid box {} for target at index {}."
        #                              .format(degen_bb, target_idx))

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        
        feature_lists = []
        for k, v in features.items():
            feature_lists.append((k, v))
        # print('feature_lists[0][1] shape: ', feature_lists[0][1].size())
        feature_to_return = F.interpolate(feature_lists[0][1], original_image_sizes[0])

        proposals, proposal_losses = self.rpn(images, features, targets)
        # return proposals, proposal_losses, feature_to_return
        self.roi_heads.training = False
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        return detections, feature_to_return

        # losses = {}
        # losses.update(detector_losses)
        # losses.update(proposal_losses)
        # return losses, detections, feature_to_return
        # # if torch.jit.is_scripting():
        # #     if not self._has_warned:
        # #         warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
        # #         self._has_warned = True
        # #     return losses, detections, feature_to_return
        # # else:
        # #     return self.eager_outputs(losses, detections), feature_to_return

if __name__ == "__main__":
    from PIL import Image
    import cv2
    import numpy as np
    import torchvision.transforms as transforms
    from skimage import io
    transform = transforms.Compose([transforms.ToTensor()])
    img_cv = io.imread('/home/newdisk/yanqiao/dataset/view-of-Delft/view_of_delft_PUBLIC/radar/training/image_2/00000.jpg')
    img_cv = img_cv.astype(np.float32)
    img_cv /= 255.0
    # img_cv_tensor = transform(img_cv)
    img_cv_tensor = torch.from_numpy(img_cv).permute(2, 0, 1)
    print(img_cv_tensor.size())
    device = torch.device("cuda:3")
    
    
        
    rcnn = RCNN().to(device)
    
    img = torch.zeros(3, 640, 480).to(device)
    img_cv_tensor = img_cv_tensor.to(device)
    
    imgs = [img_cv_tensor, img_cv_tensor]
    
    # imgs = [img, img]
    # print(img_feature.shape)
    # proposals_lists, proposal_scores_lists, feature_tensor = rcnn(imgs)
    detection, feature_tensor = rcnn(imgs)
    
    print('feature_tensor:, ', feature_tensor.size())
    print('detection:, ', len(detection))
    # print('detection:, ', detection)
    for d in detection:
        boxes = d['boxes']
        labels = d['labels']
        scores = d['scores']
        print('boxes shape: ', boxes.cpu().detach().numpy().shape)
        print('labels shape: ', labels.cpu().detach().numpy().shape)
        print('scores shape: ', scores.cpu().detach().numpy().shape)
        
    # for p,ps in zip(proposals_lists, proposal_scores_lists):
    #     print(p.cpu().detach().numpy(), ps.cpu().detach().numpy())
    #     print(p.cpu().detach().numpy().shape, ps.cpu().detach().numpy().shape)
        
    # print('proposals:, ', proposals)
    # print('proposal_scores:, ', proposal_scores)
