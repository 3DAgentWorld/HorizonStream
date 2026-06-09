
import os
import glob
import time
import threading
from typing import List, Optional, Tuple, Callable, Union

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as vt
import cv2
import matplotlib.cm as cm

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from demo_utils.visual_util import segment_sky, download_file_from_url


def apply_ema(data: np.ndarray, alpha: float) -> np.ndarray:
    """Apply exponential moving average smoothing over time."""
    if not (0 < alpha <= 1.0):
        raise ValueError("EMA alpha must be between 0 and 1")
    if data.ndim == 1:
        data = data[:, None]
    smoothed_data = np.zeros_like(data)
    smoothed_data[0] = data[0]
    for i in range(1, len(data)):
        smoothed_data[i] = alpha * data[i] + (1 - alpha) * smoothed_data[i - 1]
    return smoothed_data


def setup_camera_follow(
    server: viser.ViserServer,
    slider: viser.GuiSliderHandle,
    target_positions: np.ndarray,
    camera_positions: Optional[np.ndarray] = None,
    camera_wxyz: Optional[np.ndarray] = None,
    camera_distance: float = 2.0,
    camera_height: float = 1.0,
    camera_angle: float = -30.0,
    up_direction: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    fov: float = 45.0,
    target_ema_alpha: float = 0.05,
    camera_ema_alpha: Union[float, Callable[[], float]] = 0.05,
    frame_lag: Union[int, Callable[[], int]] = 0,
    backoff_distance: Union[float, Callable[[], float]] = 0.0,
    camera_forward: Optional[np.ndarray] = None,
) -> Tuple[Callable[[], None], Callable[[], None]]:
    """Set up camera follow behavior driven by a frame slider."""
    smoothed_target_positions = apply_ema(target_positions, target_ema_alpha)

    def _resolve_lag() -> int:
        value = frame_lag() if callable(frame_lag) else frame_lag
        return max(0, int(value))

    def _resolve_backoff() -> float:
        value = backoff_distance() if callable(backoff_distance) else backoff_distance
        return max(0.0, float(value))

    def _resolve_camera_ema_alpha() -> float:
        value = camera_ema_alpha() if callable(camera_ema_alpha) else camera_ema_alpha
        return float(np.clip(value, 1e-4, 1.0))

    if camera_positions is not None:
        if camera_wxyz is None:
            raise ValueError("camera_wxyz must be provided when camera_positions is given.")
        if len(camera_positions) != len(smoothed_target_positions) or len(camera_wxyz) != len(smoothed_target_positions):
            raise ValueError("camera_positions/camera_wxyz and target_positions must have the same length.")
        if camera_forward is not None and len(camera_forward) != len(smoothed_target_positions):
            raise ValueError("camera_forward and target_positions must have the same length.")
        # Apply EMA to explicit camera trajectory so lag mode still looks smooth.
        smoothed_camera_positions = camera_positions.copy()
        smoothed_camera_forward = None
        last_camera_ema_alpha: Optional[float] = None

        def refresh_camera_ema_if_needed():
            nonlocal smoothed_camera_positions, smoothed_camera_forward, last_camera_ema_alpha
            alpha = _resolve_camera_ema_alpha()
            if last_camera_ema_alpha is not None and np.isclose(alpha, last_camera_ema_alpha):
                return
            smoothed_camera_positions = apply_ema(camera_positions, alpha)
            if camera_forward is not None:
                smoothed_camera_forward = apply_ema(camera_forward, alpha)
                norms = np.linalg.norm(smoothed_camera_forward, axis=1, keepdims=True)
                smoothed_camera_forward = smoothed_camera_forward / np.clip(norms, 1e-8, None)
            else:
                smoothed_camera_forward = None
            last_camera_ema_alpha = alpha

        def update_camera_for_target(client: viser.ClientHandle, t: int):
            refresh_camera_ema_if_needed()
            t_follow = max(0, t - _resolve_lag())
            cam_pos = smoothed_camera_positions[t_follow].copy()
            backoff = _resolve_backoff()
            if smoothed_camera_forward is not None and backoff > 0.0:
                cam_pos = cam_pos - smoothed_camera_forward[t_follow] * backoff

            client.camera.position = cam_pos
            client.camera.wxyz = camera_wxyz[t_follow]
            client.camera.fov = np.radians(fov)
    else:
        angle_rad = np.radians(camera_angle)

        def update_camera_for_target(client: viser.ClientHandle, t: int):
            target_pos = smoothed_target_positions[t]
            cam_offset = np.array(
                [
                    -camera_distance * np.cos(angle_rad),
                    camera_height,
                    -camera_distance * np.sin(angle_rad),
                ]
            )
            if tuple(up_direction) == (0.0, 1.0, 0.0):
                final_cam_offset = np.array([cam_offset[0], cam_offset[1], cam_offset[2]])
            elif tuple(up_direction) == (0.0, 0.0, 1.0):
                final_cam_offset = np.array([cam_offset[0], cam_offset[2], cam_offset[1]])
            else:
                final_cam_offset = cam_offset

            client.camera.position = target_pos + final_cam_offset
            client.camera.look_at = target_pos
            client.camera.up_direction = up_direction
            client.camera.fov = np.radians(fov)

    original_callback: Optional[Callable] = None

    def stop_camera_follow():
        nonlocal original_callback
        if original_callback is not None:
            slider.remove_update_callback(original_callback)
            original_callback = None

    def resume_camera_follow():
        nonlocal original_callback
        if original_callback is None:
            @slider.on_update
            def callback(_):
                t = int(max(0, min(slider.value, len(smoothed_target_positions) - 1)))
                for client in server.get_clients().values():
                    update_camera_for_target(client, t)

            original_callback = callback

    return stop_camera_follow, resume_camera_follow


def viser_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,      # Low confidence percentage filter
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder_for_sky_mask: str | None = None,
    subsample: int = 1,
    video_width: int = 320,   # Video display width
    share: bool = False,
    point_size: float = 0.001,
    canonical_first_frame: bool = True,  # Use first frame as canonical (identity pose)
):
    """Visualize predictions using Viser.

    Handles multiple camera inputs (cam0, cam01, ..., cam05).
    Point clouds are placed in Frame child nodes.
    Camera 01 (and others) are rendered and synchronized with Camera 0 playback.
    """

    # ───────────────────────────── Parse Data ────────────────────────────
    img_data = {}
    xyz_data = {}
    conf_data = {}
    cam2world_data = {}
    pcd_handles = {}
    frustums = {}
    frames_roots = {}
    video_previews = {}
    gui_show_cams = {}
    
    # Store original (unsubsampled) data for online subsample adjustment
    img_data_original = {}
    xyz_data_original = {}
    conf_data_original = {}
    current_subsample = [subsample]  # Use list to allow modification in nested functions

    # Main camera (cam0)
    # Pi3 outputs images as (S,H,W,3) or (S,C,H,W) depending on processing.
    # Viser expects (H,W,3) for image previews. Let's standardize.
    # The permute in demo_viser_pi3.py results in (S, H, W, 3)
    cam0_images = pred_dict["images"]
    if cam0_images.shape[-1] != 3: # If not (S,H,W,3), assume (S,C,H,W)
         cam0_images = cam0_images.transpose(0, 2, 3, 1)

    img_data_original["cam0"] = cam0_images
    xyz_data_original["cam0"] = pred_dict["points"]
    conf_data_original["cam0"] = pred_dict["conf"]
    
    img_data["cam0"]   = cam0_images[:, ::subsample, ::subsample]
    xyz_data["cam0"]   = pred_dict["points"][:, ::subsample, ::subsample]        # (S,H,W,3)
    conf_data["cam0"]  = pred_dict["conf"][:, ::subsample, ::subsample]   # (S,H,W)
    S = xyz_data["cam0"].shape[0]

    cam_ids = ["cam0"]
    for i in range(1, 6): # Check for cam01 to cam05
        cam_id = f"cam{i:02d}"
        if cam_id in pred_dict:
            cam_ids.append(cam_id)
            
            other_cam_images = pred_dict[cam_id]["images"]
            if other_cam_images.shape[-1] != 3: # If not (S,H,W,3), assume (S,C,H,W)
                other_cam_images = other_cam_images.transpose(0, 2, 3, 1)

            img_data_original[cam_id] = other_cam_images
            xyz_data_original[cam_id] = pred_dict[cam_id]["points"]
            conf_data_original[cam_id] = pred_dict[cam_id]["conf"]
            
            img_data[cam_id]  = other_cam_images[:, :, ::subsample, ::subsample]
            xyz_data[cam_id]  = pred_dict[cam_id]["points"][:, ::subsample, ::subsample]
            conf_data[cam_id] = pred_dict[cam_id]["conf"][:, ::subsample, ::subsample]
            S = min(S, xyz_data[cam_id].shape[0]) # Unify frame count

    # Trim all data to unified frame count S
    for cam_id in cam_ids:
        img_data[cam_id] = img_data[cam_id][:S]
        xyz_data[cam_id] = xyz_data[cam_id][:S]
        conf_data[cam_id] = conf_data[cam_id][:S]
        img_data_original[cam_id] = img_data_original[cam_id][:S]
        xyz_data_original[cam_id] = xyz_data_original[cam_id][:S]
        conf_data_original[cam_id] = conf_data_original[cam_id][:S]

    # ───────────────────────────── Sky Mask ───────────────────────────
    if mask_sky and image_folder_for_sky_mask is not None:
        sky_masks_for_conf = apply_sky_segmentation(conf_data["cam0"], image_folder_for_sky_mask, is_conf_scores=True)
        for cam_id in cam_ids:
            if conf_data[cam_id].shape == sky_masks_for_conf.shape:
                 conf_data[cam_id] = conf_data[cam_id] * sky_masks_for_conf
            else:
                print(f"Warning: Shape mismatch for sky masking on {cam_id}. Skipping sky mask for this camera.")

    # ───────────────────────────── Setup Server ──────────────────────────
    server = viser.ViserServer(host="0.0.0.0", port=port)
    if share: server.request_share_url()
    server.scene.set_up_direction("-y")
    server.scene.add_frame("/frames", show_axes=False)   # Root node
    
    H_main, W_main = xyz_data["cam0"].shape[1:3]

    def process_video_frame(frame_idx, cam_id_to_process="cam0"):
        frame = img_data[cam_id_to_process][frame_idx]
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
        h, w = frame.shape[:2]
        new_w = video_width
        new_h = int(h * (new_w / w))
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized_frame

    for cam_id in cam_ids:
        video_previews[cam_id] = server.gui.add_image(
            process_video_frame(0, cam_id),
            format="jpeg",
            label=f"Camera {cam_id.replace('cam', '')}"
        )

    # ───────────── GUI – Playback ─────────────
    with server.gui.add_folder("Playback"):
        gui_play    = server.gui.add_checkbox("Playing", True)
        gui_frame   = server.gui.add_slider("Frame", 0, S-1, 1, 0, disabled=True)
        gui_next    = server.gui.add_button("Next", disabled=True)
        gui_prev    = server.gui.add_button("Prev", disabled=True)
        gui_fps     = server.gui.add_slider("FPS", 1, 60, 0.1, 20)
        gui_fps_btn = server.gui.add_button_group("FPS options", ("10","20","30","60"))
        gui_all     = server.gui.add_checkbox("Show all frames", False)
        gui_accumulate_play = server.gui.add_checkbox("Accumulate on play", False)
        gui_stride  = server.gui.add_slider("Stride", 1, S, 1, min(10, max(0, S - 1)), disabled=True)

    # ───────────── GUI – Visualization ─────────
    with server.gui.add_folder("Visualization"):
        gui_show_all_cams_master = server.gui.add_checkbox("Show All Cameras", True)
        gui_conf     = server.gui.add_slider("Confidence Percent", 0,100,0.1, init_conf_threshold)
        gui_point_size = server.gui.add_slider("Point Size", 0.0001, 0.05, 0.0001, point_size)
        gui_camera_size = server.gui.add_slider("Camera Size", 0.01, 0.3, 0.01, 0.03)
        gui_camera_follow = server.gui.add_checkbox("Camera Follow Cam0", False)
        gui_camera_follow_lag = server.gui.add_slider("Follow Lag (frames)", 0, min(30, max(0, S - 1)), 1, min(10, max(0, S - 1)))
        gui_camera_follow_backoff = server.gui.add_slider("Follow Backoff", 0.0, 3.0, 0.01, 0.25)
        gui_camera_follow_ema = server.gui.add_slider("Follow EMA Alpha", 0.001, 1.0, 0.001, 0.05)
        for cam_id in cam_ids:
            gui_show_cams[cam_id] = server.gui.add_checkbox(f"Show {cam_id.upper()}", True)
    
    # ───────────── GUI – Frame Range & Subsample ─────────
    with server.gui.add_folder("Frame Range & Subsample"):
        gui_start_frame = server.gui.add_slider("Start Frame", 0, S-1, 1, 0)
        gui_end_frame = server.gui.add_slider("End Frame", 0, S-1, 1, S-1)
        gui_subsample = server.gui.add_slider("Subsample", 1, 10, 1, subsample)
        gui_apply_range = server.gui.add_button("Apply Range & Subsample")

    # ───────────── Helper Function: Confidence Filtering ────────
    def gen_mask(conf_array, percent):
        if conf_array.size == 0 or not np.any(np.isfinite(conf_array)):
            return conf_array > -np.inf
        
        # In Pi3, conf is already a probability in [0,1]. Percentile is not the right tool.
        # We should use the percentage as a direct threshold.
        thresh = percent / 100.0
        return (conf_array >= thresh) & (conf_array > 1e-5)

    # ───────────── Create All Frame Nodes ─────────────
    # 不再进行中心化处理，直接使用原始坐标
    xyz_centered_data = {}

    # First, collect all camera poses to find the canonical transform
    T0_inv = None
    if canonical_first_frame:
        # Get the first camera pose from cam0 to use as canonical frame
        cam0_poses = pred_dict.get("camera_poses")
        if cam0_poses is not None and len(cam0_poses) > 0:
            T0 = cam0_poses[0]  # First frame's camera-to-world transform (4x4)
            if T0.shape == (4, 4):
                T0_inv = np.linalg.inv(T0)
                print("Using first frame as canonical frame (identity pose).")
            elif T0.shape == (3, 4):
                T0_full = np.eye(4)
                T0_full[:3, :] = T0
                T0_inv = np.linalg.inv(T0_full)
                print("Using first frame as canonical frame (identity pose).")

    for cam_id in cam_ids:
        pcd_handles[cam_id] = []
        frustums[cam_id] = []
        frames_roots[cam_id] = []

        # Pi3 provides camera_poses directly, assuming they are camera-to-world 4x4 matrices
        if cam_id == "cam0":
            poses = pred_dict.get("camera_poses")
        else:
            poses = pred_dict.get(cam_id, {}).get("camera_poses")

        if poses is None:
            print(f"Warning: camera_poses not found for {cam_id}. Using identity.")
            poses = np.tile(np.eye(4), (S, 1, 1))

        # Truncate poses
        poses = poses[:S]

        # Apply canonical transform if enabled
        if canonical_first_frame and T0_inv is not None:
            # Transform all poses: T'_i = T0_inv @ T_i
            transformed_poses = []
            for i in range(len(poses)):
                if poses[i].shape == (4, 4):
                    T_new = T0_inv @ poses[i]
                elif poses[i].shape == (3, 4):
                    T_full = np.eye(4)
                    T_full[:3, :] = poses[i]
                    T_new = T0_inv @ T_full
                else:
                    T_new = poses[i]
                transformed_poses.append(T_new)
            poses = np.array(transformed_poses)

        # Convert to (S, 3, 4) for viser
        if poses.shape[-2:] == (4, 4):
            cam2world_data[cam_id] = poses[:, :3, :]
        else:
            cam2world_data[cam_id] = poses

        # Transform point clouds to canonical frame if enabled
        if canonical_first_frame and T0_inv is not None:
            # Transform points: p' = R0_inv @ p + t0_inv (apply inverse of first pose)
            R0_inv = T0_inv[:3, :3]
            t0_inv = T0_inv[:3, 3]
            xyz_original = xyz_data[cam_id]  # (S, H, W, 3)
            # Reshape for matrix multiplication: (S*H*W, 3)
            original_shape = xyz_original.shape
            xyz_flat = xyz_original.reshape(-1, 3)
            # Apply rotation and translation
            xyz_transformed = (R0_inv @ xyz_flat.T).T + t0_inv
            xyz_centered_data[cam_id] = xyz_transformed.reshape(original_shape)
        else:
            # 直接使用原始点云坐标，不进行中心化
            xyz_centered_data[cam_id] = xyz_data[cam_id]

    # Camera follow uses cam0 camera trajectory over the frame slider.
    cam0_poses = cam2world_data["cam0"]  # (S, 3, 4)
    cam0_positions = cam0_poses[:, :3, 3]
    cam0_wxyz = np.array([vt.SO3.from_matrix(pose[:, :3]).wxyz for pose in cam0_poses], dtype=np.float32)
    cam0_forward = np.array([pose[:, :3] @ np.array([0.0, 0.0, 1.0]) for pose in cam0_poses], dtype=np.float32)
    cam0_lookat = cam0_positions + cam0_forward
    stop_camera_follow, resume_camera_follow = setup_camera_follow(
        server=server,
        slider=gui_frame,
        target_positions=cam0_lookat,
        camera_positions=cam0_positions,
        camera_wxyz=cam0_wxyz,
        camera_forward=cam0_forward,
        camera_ema_alpha=lambda: gui_camera_follow_ema.value,
        frame_lag=lambda: gui_camera_follow_lag.value,
        backoff_distance=lambda: gui_camera_follow_backoff.value,
        up_direction=(0.0, -1.0, 0.0),
        fov=60.0,
    )

    print("Building frames / point clouds …")
    for i in tqdm(range(S)):
        f_root_timestep = server.scene.add_frame(f"/frames/t{i}", show_axes=False)

        for cam_id in cam_ids:
            frames_roots[cam_id].append(f_root_timestep)

            # Point Cloud
            current_conf = conf_data[cam_id][i]
            current_xyz_c = xyz_centered_data[cam_id][i]
            current_img = img_data[cam_id][i]

            if gui_show_cams[cam_id].value:
                mask = gen_mask(current_conf, gui_conf.value)
                # Reshape arrays for masking: (H,W,3) -> (H*W,3) and (H,W) -> (H*W,)
                pts_flat = current_xyz_c.reshape(-1, 3)
                mask_flat = mask.reshape(-1)
                rgb_img_for_pts = current_img
                if rgb_img_for_pts.max() <= 1.0: rgb_img_for_pts = rgb_img_for_pts * 255
                rgb_flat = rgb_img_for_pts.astype(np.uint8).reshape(-1, 3)
                
                # Apply mask
                pts = pts_flat[mask_flat]
                rgb = rgb_flat[mask_flat]
            else:
                pts = np.zeros((0,3), np.float32)
                rgb = np.zeros((0,3), np.uint8)

            pcd_handle  = server.scene.add_point_cloud(
                f"/frames/t{i}/pc_{cam_id}", pts, rgb, point_size=point_size*(subsample**(1/2)), point_shape="rounded"
            )
            pcd_handles[cam_id].append(pcd_handle)

            # Frustum
            norm_i = i/(S-1) if S>1 else 0.0
            col   = cm.get_cmap('gist_rainbow')(norm_i)[:3]
            
            # Since Pi3 doesn't provide intrinsics, we have to use a heuristic for FOV.
            # This is a limitation. A fixed FOV is a reasonable fallback.
            h_img_cam, w_img_cam = img_data[cam_id].shape[-3:-1]
            # Use a more reasonable FOV - around 60 degrees (1.047 radians)
            fov_cam = 1.047  # 60 degrees in radians, typical camera FOV
            aspect_cam = w_img_cam / h_img_cam

            # Reconstruct 4x4 matrix from 3x4 for SE3
            cam_pose_3x4 = cam2world_data[cam_id][i]
            cam_pose_4x4 = np.eye(4)
            cam_pose_4x4[:3, :] = cam_pose_3x4
            T_cam = vt.SE3.from_matrix(cam_pose_4x4)
            
            # Use processed image for frustum view
            frustum_img = current_img
            if frustum_img.max() <= 1.0: frustum_img = frustum_img * 255
            frustum_img = frustum_img.astype(np.uint8)

            frustum_handle = server.scene.add_camera_frustum(
                f"/frames/t{i}/frustum_{cam_id}", fov_cam, aspect_cam, scale=gui_camera_size.value,
                image=frustum_img,
                wxyz=T_cam.rotation().wxyz, position=T_cam.translation(),
                color=col, line_width=2.0
            )
            frustums[cam_id].append(frustum_handle)

    # ───────────── Update Visibility ─────────────
    def set_visibility():
        show_all_ts = gui_all.value
        accumulate_on_play = gui_accumulate_play.value and gui_play.value and (not show_all_ts)
        stride_ts   = gui_stride.value
        current_ts  = gui_frame.value
        master_show_frustums = gui_show_all_cams_master.value
        start_f = gui_start_frame.value
        end_f = gui_end_frame.value

        for i in range(S):
            in_range = (start_f <= i <= end_f)
            if show_all_ts:
                vis_timestep_level = (i % stride_ts == 0) and in_range
            elif accumulate_on_play:
                vis_timestep_level = (start_f <= i <= current_ts) and (((i - start_f) % stride_ts) == 0) and in_range
            else:
                vis_timestep_level = (i == current_ts) and in_range
            
            if len(cam_ids) > 0 and frames_roots[cam_ids[0]][i] is not None:
                 frames_roots[cam_ids[0]][i].visible = vis_timestep_level

            for cam_id in cam_ids:
                individual_cam_active = gui_show_cams[cam_id].value

                if pcd_handles[cam_id][i] is not None:
                    pcd_handles[cam_id][i].visible = vis_timestep_level and individual_cam_active
                
                if frustums[cam_id][i] is not None:
                    frustums[cam_id][i].visible = vis_timestep_level and individual_cam_active and master_show_frustums
    
    set_visibility()

    # ───────────── Refresh Point Clouds (Confidence Slider) ─────────────
    def refresh_pointclouds():
        pct = gui_conf.value
        cur_subsample = current_subsample[0]
        new_point_size = gui_point_size.value * (cur_subsample ** 0.5)
        
        for i in tqdm(range(S), leave=False, desc="Refreshing PCs"):
            for cam_id in cam_ids:
                if gui_show_cams[cam_id].value :
                    current_conf = conf_data[cam_id][i]
                    current_xyz_c = xyz_centered_data[cam_id][i]
                    current_img = img_data[cam_id][i]

                    mask = gen_mask(current_conf, pct)
                    # Reshape arrays for masking: (H,W,3) -> (H*W,3) and (H,W) -> (H*W,)
                    pts_flat = current_xyz_c.reshape(-1, 3)
                    mask_flat = mask.reshape(-1)
                    rgb_img_for_pts = current_img
                    if rgb_img_for_pts.max() <= 1.0: rgb_img_for_pts = rgb_img_for_pts * 255
                    rgb_flat = rgb_img_for_pts.astype(np.uint8).reshape(-1, 3)
                    
                    # Apply mask
                    pts = pts_flat[mask_flat]
                    rgb = rgb_flat[mask_flat]
                    
                    pcd_handles[cam_id][i].points = pts
                    pcd_handles[cam_id][i].colors = rgb
                    pcd_handles[cam_id][i].point_size = new_point_size

    # ───────────── GUI Callback Bindings ─────────────
    @gui_next.on_click
    def _(_): gui_frame.value = (gui_frame.value+1)%S

    @gui_prev.on_click
    def _(_): gui_frame.value = (gui_frame.value-1+S)%S

    @gui_fps_btn.on_click
    def _(_): gui_fps.value = float(gui_fps_btn.value)

    @gui_play.on_update
    def _(_):
        controls_disabled = gui_play.value or gui_all.value
        gui_frame.disabled = controls_disabled
        gui_next.disabled = controls_disabled
        gui_prev.disabled = controls_disabled
        gui_stride.disabled = not (gui_all.value or gui_accumulate_play.value)
        set_visibility()

    @gui_conf.on_update
    def _(_): refresh_pointclouds()

    @gui_point_size.on_update
    def _(_):
        new_point_size = gui_point_size.value
        for cam_id in cam_ids:
            for handle in pcd_handles[cam_id]:
                if handle is not None:
                    handle.point_size = new_point_size

    @gui_camera_size.on_update
    def _(_):
        new_camera_size = gui_camera_size.value
        for cam_id in cam_ids:
            for handle in frustums[cam_id]:
                if handle is not None:
                    handle.scale = new_camera_size

    @gui_frame.on_update
    def _(_):
        set_visibility()
        current_frame_val = gui_frame.value
        for cam_id in cam_ids:
            video_previews[cam_id].image = process_video_frame(current_frame_val, cam_id)

    @gui_all.on_update
    def _(_):
        gui_stride.disabled = not (gui_all.value or gui_accumulate_play.value)
        controls_disabled = gui_play.value or gui_all.value
        gui_frame.disabled = controls_disabled
        gui_next.disabled = controls_disabled
        gui_prev.disabled = controls_disabled
        set_visibility()

    @gui_stride.on_update
    def _(_): set_visibility()

    @gui_accumulate_play.on_update
    def _(_):
        gui_stride.disabled = not (gui_all.value or gui_accumulate_play.value)
        set_visibility()

    @gui_show_all_cams_master.on_update
    def _(_):
        set_visibility()

    @gui_start_frame.on_update
    def _(_): set_visibility()
    
    @gui_end_frame.on_update
    def _(_): set_visibility()
    
    @gui_apply_range.on_click
    def _(_):
        new_subsample = gui_subsample.value
        if new_subsample != current_subsample[0]:
            print(f"Applying new subsample: {new_subsample} (was {current_subsample[0]})")
            current_subsample[0] = new_subsample
            
            # Update subsampled data
            for cam_id in cam_ids:
                img_data[cam_id] = img_data_original[cam_id][:, ::new_subsample, ::new_subsample]
                xyz_data[cam_id] = xyz_data_original[cam_id][:, ::new_subsample, ::new_subsample]
                conf_data[cam_id] = conf_data_original[cam_id][:, ::new_subsample, ::new_subsample]
                
                # Update xyz_centered_data with canonical transform
                if canonical_first_frame and T0_inv is not None:
                    R0_inv = T0_inv[:3, :3]
                    t0_inv = T0_inv[:3, 3]
                    xyz_original = xyz_data[cam_id]
                    original_shape = xyz_original.shape
                    xyz_flat = xyz_original.reshape(-1, 3)
                    xyz_transformed = (R0_inv @ xyz_flat.T).T + t0_inv
                    xyz_centered_data[cam_id] = xyz_transformed.reshape(original_shape)
                else:
                    xyz_centered_data[cam_id] = xyz_data[cam_id]
            
            # Refresh point clouds with new subsample
            refresh_pointclouds()
            print(f"Subsample updated to {new_subsample}")
        else:
            # Just refresh visibility for frame range
            set_visibility()
            print(f"Frame range applied: {gui_start_frame.value} - {gui_end_frame.value}")

    for cam_id in cam_ids:
        # Use a closure to capture the correct cam_id for the callback
        def make_callback(cam_id_captured):
            def callback(_):
                set_visibility()
                refresh_pointclouds()
            return callback
        gui_show_cams[cam_id].on_update(make_callback(cam_id))

    @gui_camera_follow.on_update
    def _(_):
        if gui_camera_follow.value:
            resume_camera_follow()
            # Apply once immediately so manual frame value is reflected.
            gui_frame.value = gui_frame.value
        else:
            stop_camera_follow()

    @gui_camera_follow_lag.on_update
    def _(_):
        if gui_camera_follow.value:
            gui_frame.value = gui_frame.value

    @gui_camera_follow_backoff.on_update
    def _(_):
        if gui_camera_follow.value:
            gui_frame.value = gui_frame.value

    @gui_camera_follow_ema.on_update
    def _(_):
        if gui_camera_follow.value:
            gui_frame.value = gui_frame.value

    # ───────────── Playback Loop ─────────────
    def loop():
        prev_time = time.time()
        while True:
            if gui_play.value and not gui_all.value:
                now = time.time()
                if now - prev_time >= 1.0/gui_fps.value:
                    start_f = gui_start_frame.value
                    end_f = gui_end_frame.value
                    next_frame = gui_frame.value + 1
                    if next_frame > end_f:
                        next_frame = start_f
                    elif next_frame < start_f:
                        next_frame = start_f
                    gui_frame.value = next_frame
                    prev_time = now
            time.sleep(0.005)

    if background_mode:
        threading.Thread(target=loop, daemon=True).start()
        print(f"Viser server running in background on port {port}")
    else:
        print(f"Viser server running in foreground on port {port}. Press Ctrl+C to stop.")
        loop()

    return server

def apply_sky_segmentation(
    data_to_mask: np.ndarray, 
    image_folder_for_sky_mask: str, 
    is_conf_scores: bool = False
) -> np.ndarray:
    """
    Apply sky segmentation. If is_conf_scores is True, `data_to_mask` are confidence scores (S, H, W)
    and the function returns a binary mask (0 for sky, 1 for non-sky) of the same shape.
    Otherwise, it assumes data_to_mask are images and directly masks them (not implemented here for that path).
    Args:
        data_to_mask (np.ndarray): Data to apply mask to, typically confidence (S, H, W) or potentially images.
        image_folder_for_sky_mask (str): Path to the folder containing original input images.
        is_conf_scores (bool): If true, data_to_mask is confidence and a binary mask is returned.
    Returns:
        np.ndarray: If is_conf_scores, returns binary non-sky mask (S,H,W). Otherwise, modifies data_to_mask (not fully implemented).
    """
    S_data, H_data, W_data = data_to_mask.shape 
    sky_masks_dir = image_folder_for_sky_mask.rstrip("/") + "_sky_masks"
    os.makedirs(sky_masks_dir, exist_ok=True)

    onnx_path = "skyseg.onnx"
    if not os.path.exists(onnx_path):
        print("Downloading skyseg.onnx...")
        try:
            download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", onnx_path)
        except Exception as e:
            print(f"Failed to download skyseg.onnx: {e}. Sky segmentation will be skipped.")
            return np.ones_like(data_to_mask) if is_conf_scores else data_to_mask

    try:
        skyseg_session = onnxruntime.InferenceSession(onnx_path)
    except Exception as e:
        print(f"Error loading ONNX model: {e}. Sky segmentation will be skipped.")
        return np.ones_like(data_to_mask) if is_conf_scores else data_to_mask
        
    try:
        from natsort import natsorted
    except ImportError:
        print("natsort library not found. File sorting may be incorrect for sky masks.")
        def natsorted(x): 
            return sorted(x)

    source_image_files = natsorted(glob.glob(os.path.join(image_folder_for_sky_mask, "*")))
    if not source_image_files:
        print(f"No images found in {image_folder_for_sky_mask} for sky segmentation. Sky segmentation skipped.")
        return np.ones_like(data_to_mask) if is_conf_scores else data_to_mask
        
    sky_mask_list = []
    print("Generating sky masks...")
    num_images_to_process = min(S_data, len(source_image_files))

    for i in tqdm(range(num_images_to_process), desc="Sky Segmentation"):
        image_path = source_image_files[i]
        image_name = os.path.basename(image_path)
        mask_filename = os.path.splitext(image_name)[0] + ".png" 
        mask_filepath = os.path.join(sky_masks_dir, mask_filename)

        if os.path.exists(mask_filepath):
            sky_mask_individual = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask_individual = segment_sky(image_path, skyseg_session, mask_filepath)

        if sky_mask_individual is None: 
            print(f"Warning: Sky mask for {image_name} could not be generated/loaded. Using no-sky mask for this frame.")
            sky_mask_individual = np.zeros((H_data, W_data), dtype=np.uint8) 

        if sky_mask_individual.shape[0] != H_data or sky_mask_individual.shape[1] != W_data:
            sky_mask_individual = cv2.resize(sky_mask_individual, (W_data, H_data), interpolation=cv2.INTER_NEAREST)
        sky_mask_list.append(sky_mask_individual)

    while len(sky_mask_list) < S_data:
        print(f"Warning: Not enough sky masks ({len(sky_mask_list)}) for all {S_data} frames/depths. Padding with no-sky masks.")
        sky_mask_list.append(np.zeros((H_data, W_data), dtype=np.uint8))

    sky_mask_array_stacked = np.array(sky_mask_list)
    
    non_sky_mask_binary = (sky_mask_array_stacked < 128).astype(np.float32)

    if is_conf_scores:
        print("Sky segmentation applied successfully (returning binary mask).")
        return non_sky_mask_binary
    else:
        print("Warning: Direct image masking in apply_sky_segmentation is not fully implemented for this path.")
        return data_to_mask 
