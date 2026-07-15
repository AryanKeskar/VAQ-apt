# Design Analysis: `gen_visualization_single.py` Failure

## Error summary

The script failed inside `src/models/entropy_utils.py` while drawing patch boundaries with OpenCV:

```
cv2.error: OpenCV(5.0.0) ... error: (-215:Assertion failed) dims <= 2 in function 'operator()'
```

This occurred at the call to `cv2.line(...)` inside `visualize_selected_patches_cv2_non_overlapping`.

## Root cause

The function `visualize_selected_patches_cv2_non_overlapping` assumes the input image tensor is a single image with one of these shapes:

- `(H, W)` for grayscale
- `(C, H, W)` for channel-first RGB/grayscale
- `(H, W, C)` for height-width-channel format

It then converts that tensor into a NumPy array and draws rectangles directly on it.

However, the actual image array passed into that function ended up with more than 3 dimensions, so `annotated_np` became a 4D array. OpenCV `cv2.line` can only draw on 2D or 3-channel 3D arrays, so it rejected the input with `dims <= 2`.

In other words, the failure is not caused by OpenCV itself, but by passing an incorrectly shaped image array into the drawing routine.

## Why it happens in this code path

The script uses the following call chain:

- `scripts/gen_visualization_single.py` -> `process_image(...)`
- inside `process_image`, it prepares `img_tensor = TF.to_tensor(img) * 255.0`
- then it calls `visualize_selected_patches_cv2_non_overlapping(image_tensor=img_tensor, ...)`

The helper function in `entropy_utils.py` contains a conversion block that only handles 2D and 3D image shapes properly. If the input tensor has an unexpected shape, for example a batched tensor like `(B, C, H, W)` or another 4D arrangement, the conversion produces a NumPy array with four dimensions.

Once `annotated_np` is 4D, the next OpenCV drawing call fails.

## Secondary warning

The script also printed a `torchvision` warning about `libjpeg.9.dylib`:

```
Failed to load image Python extension: ... Library not loaded: @rpath/libjpeg.9.dylib
```

That warning is separate from the crash. It means the `torchvision.io` image backend cannot load its native image extension, usually because `libjpeg` or `libpng` is missing from the environment. In this case the code still loaded the image successfully with PIL, so the real crash is later in the OpenCV drawing logic.

## Suggested fixes

1. Add stronger shape validation before drawing.
   - In `visualize_selected_patches_cv2_non_overlapping`, assert that `image_tensor.ndim` is 2 or 3.
   - If it is 4D with batch size 1, squeeze the batch dimension.
   - Reject unsupported shapes with a clear error message instead of allowing a bad array to reach `cv2.line`.

2. Normalize the image layout explicitly.
   - Convert all acceptable input tensors to a standard `H x W x C` NumPy array.
   - For channel-first `(C, H, W)`, permute to `(H, W, C)`.
   - For `(H, W)`, expand to `(H, W, 3)`.

3. Make the drawing function robust to both PIL and NumPy inputs.
   - If the input is already a NumPy array, verify its dimensions before copy/drawing.
   - If the input is a torch tensor with an extra batch dimension, remove it.

4. Fix the calling code if the wrong tensor is produced earlier.
   - Confirm `img_tensor` passed into the visualization function is a single image, not a batch tensor.
   - Keep `img_tensor` as shape `(C, H, W)` if that is what the visualization helper expects.

## Practical recommendation

The safest fix is to treat this as a data-shape bug in `visualize_selected_patches_cv2_non_overlapping` and make the function handle unexpected tensor dimensions gracefully. That will prevent the OpenCV call from receiving an invalid input and make the failure mode clearer in the future.
