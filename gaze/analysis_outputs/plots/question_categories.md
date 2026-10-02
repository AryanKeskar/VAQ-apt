# Question Categorization Breakdown

## 1. Explanation of Categorization Rules & Sorting Logic

The VQA questions are automatically sorted into 5 intent-based categories using keyword matching rules. The categorization logic evaluates each question text (case-insensitive) in order of precedence:

1. **Gaze & Spatial Attention**:
   - **Rule**: Contains `"looking at"` or `"where"`.
   - **Purpose**: Questions that require estimating user gaze direction, visual focus, or spatial object positioning.

2. **Action Recognition**:
   - **Rule**: Contains `"doing"` or `"action"`.
   - **Purpose**: Questions focusing on ongoing human activity or physical motion.

3. **Equipment & Tools**:
   - **Rule**: Contains `"equipment"`, `"appliance"`, `"tool"`, or `"using"`.
   - **Purpose**: Questions identifying kitchen machinery, devices, or tools being operated.

4. **Food & Objects**:
   - **Rule**: Contains `"food"`, `"ingredient"`, `"eating"`, `"filling"`, or `"container"`.
   - **Purpose**: Questions querying ingredients, food items, liquids, or storage containers.

5. **General Object QA**:
   - **Rule**: Fallback category for all remaining questions (e.g., object holding/grasping, counting, colors, general objects).

---

## 2. Category-to-Question Mapping Table

Below is the complete mapping of all 15 dataset questions grouped by their assigned category:

| Category | Sample ID | Question Text | Key Matching Trigger |
| :--- | :--- | :--- | :--- |
| **Gaze & Spatial Attention** | `P01-20240202-1613548654` | *"what is the person looking at?"* | `"looking at"` |
| **Action Recognition** | `P01-20240202-1613542471` | *"what is the person doing?"* | `"doing"` |
| **Equipment & Tools** | `P01-20240202-161354733` | *"what equipment is the person using?"* | `"equipment"`, `"using"` |
| **Equipment & Tools** | `P01-20240202-1613543281` | *"what equipment is the person using?"* | `"equipment"`, `"using"` |
| **Food & Objects** | `P01-20240202-1613547614` | *"what food is on the counter?"* | `"food"` |
| **Food & Objects** | `P01-20240202-1613541149` | *"what object is the filling?"* | `"filling"` |
| **Food & Objects** | `P01-20240202-1613549558` | *"what color is food in the pot?"* | `"food"` |
| **Food & Objects** | `P01-20240202-1613544075` | *"what is the green food in the image?"* | `"food"` |
| **Food & Objects** | `P01-20240202-1613546827` | *"what food is in the plate in front of the person?"* | `"food"` |
| **Food & Objects** | `P01-20240202-1613541446` | *"what liquid is being filled into the container?"* | `"container"` |
| **General Object QA** | `P01-20240202-1613548602` | *"what is the person holding in their right hand?"* | Fallback (Holding) |
| **General Object QA** | `P01-20240202-1613544906` | *"what is the person holding in their hand?"* | Fallback (Holding) |
| **General Object QA** | `P01-20240202-1613543131` | *"what is the person grasping in her right hand?"* | Fallback (Grasping) |
| **General Object QA** | `P01-20240202-1613545848` | *"what is the person grasping?"* | Fallback (Grasping) |
| **General Object QA** | `P01-20240202-1613548886` | *"how many pots are on the stove?"* | Fallback (Counting) |

---

## 3. Summary Count per Category

| Question Category | Total Questions | Percentage of Dataset |
| :--- | :---: | :---: |
| **Food & Objects** | 6 | 40.0% |
| **General Object QA** | 5 | 33.3% |
| **Equipment & Tools** | 2 | 13.3% |
| **Gaze & Spatial Attention** | 1 | 6.7% |
| **Action Recognition** | 1 | 6.7% |
| **Total** | **15** | **100.0%** |
