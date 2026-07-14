# Patch Tokenizer and Entropy Utilities Summary

## Overview

This pair of files supports a mixed-resolution patch tokenization pipeline for images.

- [VAQ-apt/src/models/patch_tokenizer.py](VAQ-apt/src/models/patch_tokenizer.py) defines the main `PatchTokenizer` module. It prepares image patches at multiple scales, computes importance maps, selects which patches to keep, and returns structured outputs for downstream token-based processing.
- [VAQ-apt/src/models/entropy_utils.py](VAQ-apt/src/models/entropy_utils.py) provides the supporting utilities for measuring image complexity and selecting patches. These utilities compute entropy, Laplacian, and MSE-based importance signals and can visualize the selected patch regions.

---

## 1) [VAQ-apt/src/models/patch_tokenizer.py](VAQ-apt/src/models/patch_tokenizer.py)

### `PatchTokenizer.__init__(...)`
- Purpose: Initializes the tokenizer configuration.
- Inputs:
  - `num_scales`: number of patch scales to use
  - `base_patch_size`: smallest patch size
  - `image_size`: input image size
  - `thresholds`: entropy thresholds for each scale
  - `mean`, `std`: normalization statistics
  - `method`: importance-map method (`entropy`, `laplacian`, or `upsample_mse`)
  - `laplacian_aggregate`: how Laplacian values are aggregated (`mean`, `max`, `std`)
- Output: Stores configuration and normalization transforms on the module.

### `PatchTokenizer.construct_masks(importance_maps)`
- Purpose: Converts importance maps into binary patch-selection masks.
- Inputs:
  - `importance_maps`: dictionary mapping patch sizes to importance maps of shape `(B, H_p, W_p)`
- Outputs:
  - `masks`: dictionary of per-scale binary masks
  - `output_mask`: flattened mask tensor that includes scale labels and a class-token marker `-1`
  - `seqlens`: list of selected-patch counts per batch item

### `PatchTokenizer.construct_patch_groups(images, masks, pos_embeds)`
- Purpose: Builds grouped patch tensors for each selected scale.
- Inputs:
  - `images`: input image tensor of shape `(B, C, H, W)`
  - `masks`: selection masks for each patch size
  - `pos_embeds`: optional position embeddings (currently not used in the returned output)
- Outputs:
  - dictionary containing selected patch tensors such as `full_patches_{size}` and `resized_patches_{size}`
  - also includes mask tensors for position embedding selection

### `PatchTokenizer.compute_importance_maps(images)`
- Purpose: Computes patch-importance maps for a batch of images.
- Inputs:
  - `images`: normalized image tensor of shape `(B, C, H, W)`
- Outputs:
  - dictionary mapping patch sizes to importance maps, where each map has shape `(B, H_p, W_p)`
- Notes:
  - Uses entropy, Laplacian, or MSE-based computation depending on `self.method`.

### `PatchTokenizer.forward(images, importance_maps=None, pos_embeds=None)`
- Purpose: Main entry point that runs the full tokenization pipeline.
- Inputs:
  - `images`: input image tensor of shape `(B, C, H, W)`
  - `importance_maps`: optional precomputed importance maps
  - `pos_embeds`: optional position embeddings
- Outputs:
  - dictionary containing patch groups, `output_mask`, `seqlens`, `cu_seqlens`, `max_seqlen`, and `retained_frac`
- Notes:
  - Computes importance maps if they are not provided.
  - Returns sequence metadata needed for downstream transformer-style processing.

---

## 2) [VAQ-apt/src/models/entropy_utils.py](VAQ-apt/src/models/entropy_utils.py)

### `compute_patch_entropy_vectorized(image, patch_size=16, num_scales=2, bins=512, pad_value=1e6)`
- Purpose: Computes entropy maps for one image at multiple patch sizes.
- Inputs:
  - `image`: tensor of shape `(C, H, W)` or `(H, W)` with pixel values in roughly `[0, 255]`
  - `patch_size`: base patch size
  - `num_scales`: number of scales to compute
  - `bins`: histogram bin count
  - `pad_value`: value assigned to padded boundary patches
- Output:
  - dictionary mapping patch sizes to entropy maps as tensors

### `compute_patch_entropy_batched(images, patch_size=16, num_scales=2, bins=512, pad_value=1e6)`
- Purpose: Computes entropy maps for a batch of images.
- Inputs:
  - `images`: tensor of shape `(B, C, H, W)`
- Output:
  - dictionary mapping patch sizes to batched entropy maps of shape `(B, H_p, W_p)`

### `select_patches_by_threshold(entropy_maps, thresholds, alpha=1.)`
- Purpose: Selects which image patches should be kept based on entropy thresholds.
- Inputs:
  - `entropy_maps`: dictionary of entropy maps per patch size
  - `thresholds`: list of thresholds, one per scale level
  - `alpha`: currently unused scaling factor in the function signature
- Output:
  - dictionary of masks per patch size, where values are `0/1` tensors
- Notes:
  - Larger-scale patches are selected first, then their influence is propagated to smaller scales so that overlapping regions are not double-selected.

### `visualize_selected_patches_cv2(image_tensor, masks, patch_sizes, color=(255, 255, 255), thickness=1)`
- Purpose: Draws selected patch boundaries on an image for visualization.
- Inputs:
  - `image_tensor`: image tensor in grayscale or RGB form
  - `masks`: dictionary or mapping of patch-size to mask tensors
  - `patch_sizes`: list of patch sizes to draw
  - `color`, `thickness`: drawing options
- Output:
  - a PIL image with the selected patches outlined

### `compute_patch_laplacian_vectorized(image, patch_size=16, num_scales=2, aggregate='mean', pad_mode='reflect')`
- Purpose: Computes Laplacian-based importance maps for one image at multiple scales.
- Inputs:
  - `image`: tensor of shape `(C, H, W)` or `(H, W)`
  - `patch_size`, `num_scales`, `aggregate`, `pad_mode`
- Output:
  - dictionary mapping patch sizes to Laplacian response maps

### `compute_patch_laplacian_batched(images, patch_size=16, num_scales=2, aggregate='mean', pad_mode='reflect')`
- Purpose: Computes Laplacian-based importance maps for a batch of images.
- Inputs:
  - `images`: tensor of shape `(B, C, H, W)`
- Output:
  - dictionary mapping patch sizes to batched Laplacian maps of shape `(B, H_p, W_p)`

### `compute_patch_mse_batched(images, patch_size=16, num_scales=3, scale_factors=None, aggregate='mean')`
- Purpose: Computes MSE-based importance maps by comparing each image against a downsampled-and-upsampled reconstruction.
- Inputs:
  - `images`: tensor of shape `(B, C, H, W)`
  - `patch_size`, `num_scales`, `scale_factors`, `aggregate`
- Output:
  - dictionary mapping patch sizes to MSE maps of shape `(B, H_p, W_p)`

### `visualize_selected_patches_cv2_non_overlapping(...)`
- Purpose: Similar to the previous visualization helper, but avoids drawing overlapping boundaries by tracking edges.
- Inputs:
  - `image_tensor`, `masks`, `patch_sizes`, `color`, `thickness`
- Output:
  - a PIL image with non-overlapping patch outlines
