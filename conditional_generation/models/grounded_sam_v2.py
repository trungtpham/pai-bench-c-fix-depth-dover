"""Grounded SAM model."""

import logging
import pathlib
import sys
from collections import OrderedDict
from typing import List, Tuple

import imageio
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

from schemas import eff_segmentation
from utils import model_utils, tmp_files

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.resolve()
_GROUNDED_SAM2_PATH = PROJECT_ROOT / "third_party/Grounded-SAM-2"

# pyright: reportMissingImports=false
# pyright: reportAttributeAccessIssue=false
import os
sys.path.append(_GROUNDED_SAM2_PATH)
import hydra
import sam2 as _sam2_pkg
from hydra import initialize_config_dir
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# Config dir resolution — mirrors imaginaire4/grounded_sam_v2.py:
#   1. Prefer <grounded-sam2>/sam2_configs/ (has both sam2 and sam2.1 configs)
#   2. Fall back to the installed sam2 package root (sam2_hiera_l.yaml is there)
_sam2_configs_dir = str(_GROUNDED_SAM2_PATH / "sam2_configs")
if not os.path.isdir(_sam2_configs_dir):
    _sam2_configs_dir = str(pathlib.Path(_sam2_pkg.__file__).parent)
hydra.core.global_hydra.GlobalHydra.instance().clear()
initialize_config_dir(_sam2_configs_dir)


GROUNDING_DINO_HF_WEIGHTS_NAME = "IDEA-Research/grounding-dino-tiny"
SAM_V2_WEIGHTS_NAME = "sam2"
SAM_V2_WEIGHTS_FILENAME = "sam2_hiera_large"
SAM_V2_MODEL_CFG = "sam2_hiera_l"
# Predict classes and hyper-param for GroundingDINO
_BOX_THRESHOLD = 0.3
_TEXT_THRESHOLD = 0.25

IMAGE_MEAN = np.array([0.485, 0.456, 0.406])
IMAGE_STD = np.array([0.229, 0.224, 0.225])
NUM_POINTS_SAMPLED = 10
# Hard cap on tracked objects per video. GroundingDINO over-detects on rich
# PAI-Bench-C prompts (up to 42 objects observed), causing SAM2 to propagate
# too many tracks through 121 frames and trigger CUDA OOM (SIGABRT). The cap
# keeps the top-N highest-confidence detections (DINO output is confidence-sorted).
MAX_SAM_OBJECTS_PER_VIDEO = 20


def sample_points_from_masks(masks, num_points):
    """
    sample points from masks and return its absolute coordinates

    Args:
        masks: np.array with shape (n, h, w)
        num_points: int

    Returns:
        points: np.array with shape (n, points, 2)
    """
    n, h, w = masks.shape
    points = []

    for i in range(n):
        # find the valid mask points
        indices = np.argwhere(masks[i] == 1)
        # the output format of np.argwhere is (y, x) and the shape is (num_points, 2)
        # we should convert it to (x, y)
        indices = indices[:, ::-1]  # (num_points, [y x]) to (num_points, [x y])

        # import pdb; pdb.set_trace()
        if len(indices) == 0:
            # if there are no valid points, append an empty array
            points.append(np.array([]))
            continue

        # Deterministic sampling: seed from mask content so same mask → same points.
        # Using the global np.random is non-deterministic across pipeline runs and
        # causes recall variance of ~1/N_objects between otherwise identical pipelines.
        rng = np.random.default_rng(seed=int(indices.sum()) & 0x7FFFFFFF)
        if len(indices) < num_points:
            sampled_indices = rng.choice(len(indices), num_points, replace=True)
        else:
            sampled_indices = rng.choice(len(indices), num_points, replace=False)

        sampled_points = indices[sampled_indices]
        points.append(sampled_points)

    # convert to np.array
    points = np.array(points, dtype=np.float32)
    return points


def pack2tensor(
    images: torch.Tensor,
    video_height: int,
    video_width: int,
    offload_video_to_cpu: bool = False,
    offload_state_to_cpu: bool = False,
) -> dict:
    """
    Pack the video frames into a tensor for the model.
    images: torch.Tensor
        The frames of the video.
    video_height: int
        The height of the video.
    video_width: int
        The width of the video.
    offload_video_to_cpu: bool
        Whether to offload the video frames to CPU memory.
    offload_state_to_cpu: bool
        Whether to offload the inference state to CPU memory.

    output: dict
        The packed tensor.
    """
    inference_state = {}
    inference_state["images"] = images
    inference_state["num_frames"] = len(images)
    # whether to offload the video frames to CPU memory
    # turning on this option saves the GPU memory with only a very small overhead
    inference_state["offload_video_to_cpu"] = offload_video_to_cpu
    # whether to offload the inference state to CPU memory
    # turning on this option saves the GPU memory at the cost of a lower tracking fps
    # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
    # and from 24 to 21 when tracking two objects)
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    # the original video height and width, used for resizing final output scores
    inference_state["video_height"] = video_height
    inference_state["video_width"] = video_width
    inference_state["device"] = torch.device("cuda")
    if offload_state_to_cpu:
        inference_state["storage_device"] = torch.device("cpu")
    else:
        inference_state["storage_device"] = torch.device("cuda")
    # inputs on each frame
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    # visual features on a small number of recently visited frames for quick interactions
    inference_state["cached_features"] = {}
    # values that don't change across frames (so we only need to hold one copy of them)
    inference_state["constants"] = {}
    # mapping between client-side object id and model-side object index
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    # A storage to hold the model's tracking results and states on each frame
    inference_state["output_dict"] = {
        "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
    }
    # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
    inference_state["output_dict_per_obj"] = {}
    # A temporary storage to hold new outputs when user interact with a frame
    # to add clicks or mask (it's merged into "output_dict" before propagation starts)
    inference_state["temp_output_dict_per_obj"] = {}
    # Frames that already holds consolidated outputs from click or mask inputs
    # (we directly use their consolidated outputs during tracking)
    inference_state["consolidated_frame_inds"] = {
        "cond_frame_outputs": set(),  # set containing frame indices
        "non_cond_frame_outputs": set(),  # set containing frame indices
    }
    # metadata for each tracking frame (e.g. which direction it's tracked)
    inference_state["tracking_has_started"] = False
    inference_state["frames_already_tracked"] = {}
    inference_state["frames_tracked_per_obj"] = {}  # required by sam2>=1.1.0
    # Warm up the visual backbone and cache the image feature on frame 0

    """
    we need to return dictionary here since the function is copied from the original code.
    Changing the return type will require changing the original code significantly.
    """
    return inference_state


class GroundedSAMV2(model_utils.ModelInterface):
    """Model for grounded SAMv2.

    We do something a bit weird here. Conceptually grounded SAMv2 is a single algorithm, however it's actually two
    seperate models. The algorithm is basically:

    - Run grounding DINO and get a bunch of detected bounding boxes
    - Apply non-minumum suppression to filter out some boxes
    - Apply segment anything to every remaining detection
    - Apply segmentation v2 to track segementation mask in the videos

    The first two steps are done by the grounding DINO model, and the last two steps are done by the SAMv2 model.
    """

    @property
    def weights_names(self) -> List[str]:
        return [GROUNDING_DINO_HF_WEIGHTS_NAME, SAM_V2_WEIGHTS_NAME]

    @property
    def conda_env_name(self) -> str:
        return "paibench-transer"

    def setup(self) -> None:
        self.download_weights()
        # set up sam v2
        sam_v2_model_dir = model_utils.get_local_dir_for_weights_name(
            SAM_V2_WEIGHTS_NAME
        )
        sam_v2_weights_filepath = sam_v2_model_dir / (SAM_V2_WEIGHTS_FILENAME + ".pt")
        print(sam_v2_weights_filepath, "<sam_v2_weights_filepath>")
        sam_v2_model_cfg = SAM_V2_MODEL_CFG + ".yaml"

        self.video_predictor = build_sam2_video_predictor(
            str(sam_v2_model_cfg), str(sam_v2_weights_filepath)
        )
        sam2_image_model = build_sam2(
            str(sam_v2_model_cfg), str(sam_v2_weights_filepath)
        )
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)

        local_dir = model_utils.get_local_dir_for_weights_name(
            GROUNDING_DINO_HF_WEIGHTS_NAME
        )
        self.grounding_dino_processor = AutoProcessor.from_pretrained(local_dir)
        self.grounding_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            local_dir
        ).to(model_utils.CUDA_DEVICE)

    @property
    def conda_env_name(self) -> str:
        return "paibench-conditional-generation"

    def _read_frames(
        self,
        video_path: str,
        image_size: int = 1024,
        img_mean: np.ndarray = IMAGE_MEAN,
        img_std: np.ndarray = IMAGE_STD,
    ) -> Tuple[np.ndarray, list[np.ndarray], Tuple[int, int]]:
        """
        Read the frames of the video and preprocess them for the model.
        video_path: str
            The path to the video file.
        image_size: int
            The size of the image to resize the frames to.
        img_mean: np.ndarray
            The mean to normalize the images.
        img_std: np.ndarray
            The standard deviation to normalize the images.

        output: Tuple[np.ndarray, List[np.ndarray], Tuple[int, int]]
            The preprocessed frames, the original frames, and the original shape of the frames.
        """

        reader = imageio.get_reader(video_path)

        # Iterate over the frames of the video
        frames = []
        ori_frames = []
        for _, frame in enumerate(reader):  # type: ignore
            ori_frames.append(frame)
            resized_frame = np.array(
                Image.fromarray(frame).resize((image_size, image_size))
            )

            normalized_frame = (resized_frame / 255 - img_mean[None, None]) / img_std[
                None, None
            ]
            frames.append(normalized_frame.transpose((2, 0, 1)))

        frames = np.stack(frames)
        h, w, _ = ori_frames[0].shape

        return frames, ori_frames, (h, w)

    def _ground_dino_predict_one_image(
        self, image: np.ndarray, caption: str, image_shape: Tuple[int, int]
    ) -> dict:
        """
        Predict objects in one image using grounding dino.
        image: np.ndarray
            The image to predict the objects in.
        caption: str
            The caption to predict the objects with.
        image_shape: Tuple
            The shape of the image.

        output: dict
            The predicted objects.
        """
        if isinstance(image, np.ndarray) and not image.data.c_contiguous:
            image = np.ascontiguousarray(image)
        inputs = self.grounding_dino_processor(
            images=image,
            text=caption,
            return_tensors="pt",
            max_length=256,
        ).to(model_utils.CUDA_DEVICE)

        with torch.no_grad():
            outputs = self.grounding_dino_model(**inputs)

        results = self.grounding_dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=_BOX_THRESHOLD,
            text_threshold=_TEXT_THRESHOLD,
            target_sizes=[image_shape],
        )
        return results

    def _sam_v2_predict_one_image(
        self, image: np.ndarray, input_boxes: list
    ) -> np.ndarray:
        """
        Predict segmentation masks for one image using SAMv2.
        image: np.ndarray
            The image to predict the segmentation masks for.
        input_boxes: List
            The input boxes for the objects.

        output: np.ndarray
            The predicted segmentation masks. Each mask is a boolean array.
        """
        # predict segmentation mask for one image
        self.image_predictor.set_image(image)

        # prompt SAM 2 image predictor to get the mask for the object
        masks, _, _ = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        # convert the mask shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        return masks

    def _sam_v2_predict_video(
        self,
        frames: np.ndarray,
        original_shape: Tuple,
        image_masks: np.ndarray,
        objects: list,
    ) -> list[dict]:
        """
        Predict segmentation masks for the video using SAMv2.
        frames: np.ndarray
            The frames of the video.
        original_shape: Tuple
            The original shape of the frames.
        image_masks: np.ndarray
            The segmentation masks for the objects in the first frame.
        objects: List
            The objects in the first frame.

        output: List[dict]
            The predicted segmentation masks for the video.
            Each dict contains the segmentation masks for each object of the current frame.
        """
        frames_torch = torch.from_numpy(frames)

        inference_state = pack2tensor(
            frames_torch,
            original_shape[0],
            original_shape[1],
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
        self.video_predictor._get_image_feature(
            inference_state, frame_idx=0, batch_size=1
        )

        all_sample_points = sample_points_from_masks(
            masks=image_masks, num_points=NUM_POINTS_SAMPLED
        )

        for object_id, (_, points) in enumerate(
            zip(objects, all_sample_points), start=1
        ):
            labels = np.ones((points.shape[0]), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=object_id,
                points=points,
                labels=labels,
            )

        video_segments = (
            []
        )  # video_segments contains the per-frame segmentation results
        for (
            _,
            out_obj_ids,
            out_mask_logits,
        ) in self.video_predictor.propagate_in_video(inference_state):
            video_segment = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            video_segments.append(video_segment)

        return video_segments

    def generate_single(
        self, video_path_or_bytes: "str | bytes", caption: str
    ) -> list[eff_segmentation.SAMV2Detection]:
        """
        Generate segmentation masks for a single video. The video is processed frame by frame.
        We first run grounding DINO to detect objects in the first frame.
        Then we run SAMv2 to generate the segmentation masks for the first frame.
        Finally, we propagate the segmentation masks to the rest of the frames.
        video_path_or_bytes: str | bytes
            Path to the video file on disk (preferred — avoids /tmp roundtrip) or raw video bytes.
        caption: str
            The caption to generate the segmentation masks with.

        output: List[eff_segmentation.SAMV2Detection]
            The generated segmentation masks.
            Each detection contains the phrase and the segmentation masklet of the whole video.
            The segmentation masklet is a boolean array of shape (T, H, W), where T is the number of frames.
            WE use RLE encoding to compress the masklet.
        """

        if isinstance(video_path_or_bytes, (str, pathlib.Path)):
            frames_norm, frames, original_shape = self._read_frames(str(video_path_or_bytes))
        else:
            # Fallback: write bytes to a temp file then read — avoid /tmp by using
            # the same filesystem as the workspace to prevent ffmpeg open failures
            # under concurrent load.
            import os, tempfile
            tmpdir = os.environ.get("PAIBENCH_TMPDIR") or os.environ.get("TMPDIR") or tempfile.gettempdir()
            with tmp_files.make_named_temporary_file(suffix=".mp4", target_dir=pathlib.Path(tmpdir)) as tmp_file:
                tmp_file.write_bytes(video_path_or_bytes)
                frames_norm, frames, original_shape = self._read_frames(str(tmp_file))

        # handle empty video
        if len(frames) == 0:
            return []

        results = self._ground_dino_predict_one_image(
            frames[0][:, :, ::-1], caption, original_shape
        )

        input_boxes = results[0]["boxes"].cpu().numpy()
        # handle empty boxes
        if input_boxes.shape[0] == 0:
            return []
        objects = results[0]["labels"]

        """
        Wired design that we pass raw first frame for image segmentation and normalized images in video segmentation.
        The reason is SAMv2 reads raw image instead of mp4 video.
        Since I don't want to change the original code, I have to read videos here and do different operations.
        """
        image_masks = self._sam_v2_predict_one_image(frames[0], input_boxes)

        new_objects = []
        new_image_masks = []
        for i, image_mask in enumerate(image_masks):
            if image_mask.sum() > NUM_POINTS_SAMPLED:
                new_objects.append(objects[i])
                new_image_masks.append(image_mask)
        objects = new_objects
        if len(objects) == 0:
            return []

        if len(objects) > MAX_SAM_OBJECTS_PER_VIDEO:
            logger.warning(
                f"Capping SAM tracks from {len(objects)} to {MAX_SAM_OBJECTS_PER_VIDEO} "
                f"(DINO over-detected on rich caption)"
            )
            objects = objects[:MAX_SAM_OBJECTS_PER_VIDEO]
            new_image_masks = new_image_masks[:MAX_SAM_OBJECTS_PER_VIDEO]

        image_masks = np.array(new_image_masks)

        video_segments = self._sam_v2_predict_video(
            frames_norm, original_shape, image_masks, new_objects
        )

        """
        Step 4: Propagate the video predictor to get the segmentation results for each frame
        """

        id_to_objects = {i: obj for i, obj in enumerate(objects, start=1)}
        video_masks = {i: [] for i in id_to_objects.keys()}
        for segments in video_segments:
            for obj_id, masks in video_masks.items():
                if obj_id in segments:
                    masks.append(segments[obj_id])
                else:
                    masks.append(np.zeros((1, *original_shape)).astype(bool))

        detections = []
        for obj_id, masks in video_masks.items():
            mask = np.concatenate(masks)
            phrase = id_to_objects[obj_id]
            if phrase != "":
                item = eff_segmentation.SAMV2Detection(
                    phrase=phrase,  # type: ignore
                    segmentation_mask_rle=eff_segmentation.RleMaskSAMv2.encode(mask),  # type: ignore
                )
                detections.append(item)

        return detections


def main() -> None:
    model_utils.push_huggingface_model_to_pbss(GROUNDING_DINO_HF_WEIGHTS_NAME)


if __name__ == "__main__":
    main()
