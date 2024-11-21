# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


**Diagnostics Output:**

MAP
Traceback (most recent call last):
  File "C:\Users\mattz\Downloads\Projects\workspace\mod3-mattz17\project\parallel_check.py", line 11, in <module>
    print(tmap.parallel_diagnostics(level=3)) # type: ignore
          ^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'function' object has no attribute 'parallel_diagnostics'
(.venv) (base) PS C:\Users\mattz\Downloads\Projects\workspace\mod3-mattz17> $env:NUMBA_DISABLE_JIT = "0"
(.venv) (base) PS C:\Users\mattz\Downloads\Projects\workspace\mod3-mattz17> python project/parallel_check.py
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
C:\Users\mattz\Downloads\Projects\workspace\mod3-mattz17\minitorch\fast_ops.py
(177)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, C:\Users\mattz\Downloads\Projects\workspace\mod3-mattz17\minitorch\fast_ops.py (177)
-------------------------------------------------------------------------|loop #ID
    def _map(                                                            |
        out: Storage,                                                    |
        out_shape: Shape,                                                |
        out_strides: Strides,                                            |
        in_storage: Storage,                                             |
        in_shape: Shape,                                                 |
        in_strides: Strides,                                             |
    ) -> None:                                                           |
        # TODO: Implement for Task 3.1.                                  |
        # Check if stride alignment optimization can be applied          |
        if (                                                             |
            len(in_strides) == len(out_strides)                          |
            and len(in_shape) == len(out_shape)                          |
            and (in_strides == out_strides).all()------------------------| #0
            and (in_shape == out_shape).all()----------------------------| #1
        ):                                                               |
            for i in prange(out.size):-----------------------------------| #4
                out[i] = fn(in_storage[i])                               |
        else:                                                            |
            # Use explicit indexing for non-aligned strides              |
                                                                         |
            for i in prange(out.size):-----------------------------------| #5
                out_idx = np.zeros(len(out_shape), dtype=np.int32)-------| #2
                in_idx = np.zeros(len(in_shape), dtype=np.int32)---------| #3
                # Compute the multi-dimensional indices for output       |
                to_index(i, out_shape, out_idx)                          |
                broadcast_index(out_idx, out_shape, in_shape, in_idx)    |
                in_pos = index_to_position(in_idx, in_strides)           |
                out_pos = index_to_position(out_idx, out_strides)        |
                out[out_pos] = fn(in_storage[in_pos])                    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 6 parallel for-
loop(s) (originating from loops labelled: #0, #1, #4, #5, #2, #3).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--5 is a parallel loop
   +--2 --> rewritten as a serial loop
   +--3 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--5 (parallel)
   +--2 (parallel)
   +--3 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--5 (parallel)
   +--2 (serial)
   +--3 (serial)



Parallel region 0 (loop #5) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#5).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
C:\Users\mattz\Downloads\Projects\workspace\mod3-mattz17\minitorch\fast_ops.py
(199) is hoisted out of the parallel loop labelled #5 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_idx = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\mattz\Downloads\Projects\workspace\mod3-mattz17\minitorch\fast_ops.py
(200) is hoisted out of the parallel loop labelled #5 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: in_idx = np.zeros(len(in_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
C:\Users\mattz\Downloads\Projects\workspace\mod3-mattz17\minitorch\fast_ops.py
(234)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, C:\Users\mattz\Downloads\Projects\workspace\mod3-mattz17\minitorch\fast_ops.py (234)
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              |
        out: Storage,                                                      |
        out_shape: Shape,                                                  |
        out_strides: Strides,                                              |
        a_storage: Storage,                                                |
        a_shape: Shape,                                                    |
        a_strides: Strides,                                                |
        b_storage: Storage,                                                |
        b_shape: Shape,                                                    |
        b_strides: Strides,                                                |
    ) -> None:                                                             |
        # TODO: Implement for Task 3.1.                                    |
        stride_check = (                                                   |
            len(a_strides) == len(b_strides) == len(out_strides)           |
            and (a_strides == out_strides).all()---------------------------| #6
            and (b_strides == out_strides).all()---------------------------| #7
        )                                                                  |
        shape_check = (                                                    |
            len(a_shape) == len(b_shape) == len(out_shape)                 |
            and (a_shape == out_shape).all()-------------------------------| #8
            and (b_shape == out_shape).all()-------------------------------| #9
        )                                                                  |
        if stride_check and shape_check:                                   |
            for i in prange(out.size):-------------------------------------| #10
                out[i] = fn(a_storage[i], b_storage[i])                    |
        else:                                                              |
            # Iterate over all elements using the broadcasted shape        |
            for ordinal in prange(out.size):-------------------------------| #11
                out_index = np.empty(len(out_shape), dtype=np.int32)       |
                a_index = np.empty(len(a_shape), dtype=np.int32)           |
                b_index = np.empty(len(b_shape), dtype=np.int32)           |
                # Convert the ordinal to a multidimensional index          |
                to_index(ordinal, out_shape, out_index)                    |
                                                                           |
                # Broadcast the indices for `a` and `b`                    |
                broadcast_index(out_index, out_shape, a_shape, a_index)    |
                broadcast_index(out_index, out_shape, b_shape, b_index)    |
                                                                           |
                # Compute the storage positions                            |
                out_pos = index_to_position(out_index, out_strides)        |
                a_pos = index_to_position(a_index, a_strides)              |
                b_pos = index_to_position(b_index, b_strides)              |
                                                                           |
                # Apply the function and store the result                  |
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])      |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 6 parallel for-
loop(s) (originating from loops labelled: #6, #7, #8, #9, #10, #11).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
C:\Users\mattz\Downloads\Projects\workspace\mod3-mattz17\minitorch\fast_ops.py
(262) is hoisted out of the parallel loop labelled #11 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\mattz\Downloads\Projects\workspace\mod3-mattz17\minitorch\fast_ops.py
(263) is hoisted out of the parallel loop labelled #11 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: a_index = np.empty(len(a_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\mattz\Downloads\Projects\workspace\mod3-mattz17\minitorch\fast_ops.py
(264) is hoisted out of the parallel loop labelled #11 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: b_index = np.empty(len(b_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
C:\Users\mattz\Downloads\Projects\workspace\mod3-mattz17\minitorch\fast_ops.py
(304)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, C:\Users\mattz\Downloads\Projects\workspace\mod3-mattz17\minitorch\fast_ops.py (304)
------------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                    |
        out: Storage,                                                               |
        out_shape: Shape,                                                           |
        out_strides: Strides,                                                       |
        a_storage: Storage,                                                         |
        a_shape: Shape,                                                             |
        a_strides: Strides,                                                         |
        reduce_dim: int,                                                            |
    ) -> None:                                                                      |
        # TODO: Implement for Task 3.1.                                             |
        # if (                                                                      |
        #     len(out_shape) + 1 == len(a_shape)                                    |
        #     and a_strides[:reduce_dim] == out_strides[:reduce_dim]                |
        #     and a_strides[reduce_dim + 1:] == out_strides[reduce_dim:]            |
        # ):                                                                        |
        for i in prange(len(out)):--------------------------------------------------| #12
            out_index = np.empty(len(out_shape), dtype=np.int32)                    |
            a_index = np.empty(len(a_shape), dtype=np.int32)                        |
            to_index(i, out_shape, out_index)                                       |
            to_index(i, out_shape, a_index)                                         |
            a_index[reduce_dim] = 0                                                 |
            a_pos = index_to_position(a_index, a_strides)                           |
            reduce_res = a_storage[a_pos]                                           |
                                                                                    |
            for j in range(1, a_shape[reduce_dim]):                                 |
                a_index[reduce_dim] = j                                             |
                reduce_res = fn(                                                    |
                    reduce_res, a_storage[index_to_position(a_index, a_strides)]    |
                )                                                                   |
            out[index_to_position(out_index, out_strides)] = reduce_res             |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #12).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
C:\Users\mattz\Downloads\Projects\workspace\mod3-mattz17\minitorch\fast_ops.py
(320) is hoisted out of the parallel loop labelled #12 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\mattz\Downloads\Projects\workspace\mod3-mattz17\minitorch\fast_ops.py
(321) is hoisted out of the parallel loop labelled #12 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: a_index = np.empty(len(a_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
C:\Users\mattz\Downloads\Projects\workspace\mod3-mattz17\minitorch\fast_ops.py
(338)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, C:\Users\mattz\Downloads\Projects\workspace\mod3-mattz17\minitorch\fast_ops.py (338)
-----------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                             |
    out: Storage,                                                                        |
    out_shape: Shape,                                                                    |
    out_strides: Strides,                                                                |
    a_storage: Storage,                                                                  |
    a_shape: Shape,                                                                      |
    a_strides: Strides,                                                                  |
    b_storage: Storage,                                                                  |
    b_shape: Shape,                                                                      |
    b_strides: Strides,                                                                  |
) -> None:                                                                               |
    """NUMBA tensor matrix multiply function.                                            |
                                                                                         |
    Should work for any tensor shapes that broadcast as long as                          |
                                                                                         |
    ```                                                                                  |
    assert a_shape[-1] == b_shape[-2]                                                    |
    ```                                                                                  |
                                                                                         |
    Optimizations:                                                                       |
                                                                                         |
    * Outer loop in parallel                                                             |
    * No index buffers or function calls                                                 |
    * Inner loop should have no global writes, 1 multiply.                               |
                                                                                         |
                                                                                         |
    Args:                                                                                |
    ----                                                                                 |
        out (Storage): storage for `out` tensor                                          |
        out_shape (Shape): shape for `out` tensor                                        |
        out_strides (Strides): strides for `out` tensor                                  |
        a_storage (Storage): storage for `a` tensor                                      |
        a_shape (Shape): shape for `a` tensor                                            |
        a_strides (Strides): strides for `a` tensor                                      |
        b_storage (Storage): storage for `b` tensor                                      |
        b_shape (Shape): shape for `b` tensor                                            |
        b_strides (Strides): strides for `b` tensor                                      |
                                                                                         |
    Returns:                                                                             |
    -------                                                                              |
        None : Fills in `out`                                                            |
                                                                                         |
    """                                                                                  |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                               |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                               |
                                                                                         |
    # TODO: Implement for Task 3.2.                                                      |
    batch_size = max(a_shape[0], b_shape[0], out_shape[0])                               |
    M = out_shape[-2]                                                                    |
    N = out_shape[-1]                                                                    |
    K = a_shape[-1]                                                                      |
    out_batch_stride = out_strides[0] if out_shape[0] > 1 else 0                         |
    for batch in prange(batch_size):-----------------------------------------------------| #13
        # Compute base offsets for the current batch                                     |
        a_batch_offset = batch * a_batch_stride                                          |
        b_batch_offset = batch * b_batch_stride                                          |
        out_batch_offset = batch * out_batch_stride                                      |
                                                                                         |
        # Loop over output matrix dimensions (M, N)                                      |
        for i in range(M):                                                               |
            for j in range(N):                                                           |
                # Initialize the output value                                            |
                sum_value = 0.0                                                          |
                                                                                         |
                # Compute the dot product for the (i, j) element                         |
                for k in range(K):                                                       |
                    a_index = a_batch_offset + i * a_strides[-2] + k * a_strides[-1]     |
                    b_index = b_batch_offset + k * b_strides[-2] + j * b_strides[-1]     |
                    sum_value += a_storage[a_index] * b_storage[b_index]                 |
                                                                                         |
                # Store the result                                                       |
                out[out_batch_offset + i * out_strides[-2] + j * out_strides[-1]] = (    |
                    sum_value                                                            |
                )                                                                        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #13).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None



**Timing Script:**

Timing summary
Size: 64
    fast: 0.00341
    gpu: 0.00638
Size: 128
    fast: 0.01583
    gpu: 0.01476
Size: 256
    fast: 0.09754
    gpu: 0.05336
Size: 512
    fast: 1.21409
    gpu: 0.33832
Size: 1024
    fast: 8.39757
    gpu: 0.98560

![image](https://github.com/user-attachments/assets/fdaf7ff4-0778-4ff9-8b5a-4ab43329da87)

**Simple CPU:**

!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05


Epoch 10, loss: 1.0975, correct: 50, time per epoch: 0.1278 seconds
Epoch 20, loss: 0.2658, correct: 49, time per epoch: 0.2733 seconds
Epoch 30, loss: 1.0987, correct: 50, time per epoch: 0.1235 seconds
Epoch 40, loss: 1.5462, correct: 48, time per epoch: 0.1244 seconds
Epoch 50, loss: 0.9800, correct: 49, time per epoch: 0.1340 seconds
Epoch 60, loss: 0.3190, correct: 49, time per epoch: 0.1263 seconds
Epoch 70, loss: 0.2950, correct: 50, time per epoch: 0.1242 seconds
Epoch 80, loss: 0.3094, correct: 50, time per epoch: 0.1272 seconds
Epoch 90, loss: 0.2909, correct: 49, time per epoch: 0.1256 seconds
Epoch 100, loss: 0.4321, correct: 49, time per epoch: 0.1269 seconds
Epoch 110, loss: 0.2355, correct: 49, time per epoch: 0.1265 seconds
Epoch 120, loss: 1.5388, correct: 50, time per epoch: 0.2406 seconds
Epoch 130, loss: 0.4749, correct: 50, time per epoch: 0.1237 seconds
Epoch 140, loss: 0.4089, correct: 49, time per epoch: 0.1281 seconds
Epoch 150, loss: 0.7719, correct: 50, time per epoch: 0.1348 seconds
Epoch 160, loss: 0.1960, correct: 49, time per epoch: 0.1258 seconds
Epoch 170, loss: 0.1421, correct: 50, time per epoch: 0.1257 seconds
Epoch 180, loss: 0.0552, correct: 50, time per epoch: 0.1285 seconds
Epoch 190, loss: 0.4472, correct: 49, time per epoch: 0.1259 seconds
Epoch 200, loss: 1.0126, correct: 49, time per epoch: 0.1317 seconds
Epoch 210, loss: 0.6943, correct: 49, time per epoch: 0.2713 seconds
Epoch 220, loss: 0.4312, correct: 50, time per epoch: 0.1264 seconds
Epoch 230, loss: 0.1452, correct: 50, time per epoch: 0.1255 seconds
Epoch 240, loss: 0.9596, correct: 50, time per epoch: 0.1258 seconds
Epoch 250, loss: 0.4213, correct: 50, time per epoch: 0.1387 seconds
Epoch 260, loss: 0.2358, correct: 50, time per epoch: 0.1280 seconds
Epoch 270, loss: 0.0294, correct: 50, time per epoch: 0.1277 seconds
Epoch 280, loss: 0.9813, correct: 49, time per epoch: 0.1272 seconds
Epoch 290, loss: 0.0824, correct: 50, time per epoch: 0.1278 seconds
Epoch 300, loss: 0.0390, correct: 50, time per epoch: 0.1693 seconds
Epoch 310, loss: 0.6288, correct: 50, time per epoch: 0.1686 seconds
Epoch 320, loss: 0.7274, correct: 50, time per epoch: 0.1263 seconds
Epoch 330, loss: 0.3867, correct: 50, time per epoch: 0.1246 seconds
Epoch 340, loss: 0.8411, correct: 50, time per epoch: 0.1248 seconds
Epoch 350, loss: 0.0384, correct: 50, time per epoch: 0.1374 seconds
Epoch 360, loss: 0.1214, correct: 50, time per epoch: 0.1244 seconds
Epoch 370, loss: 0.7356, correct: 50, time per epoch: 0.1295 seconds
Epoch 380, loss: 0.0003, correct: 50, time per epoch: 0.1261 seconds
Epoch 390, loss: 0.0987, correct: 50, time per epoch: 0.1254 seconds
Epoch 400, loss: 0.0009, correct: 50, time per epoch: 0.2365 seconds
Epoch 410, loss: 0.0579, correct: 50, time per epoch: 0.1269 seconds
Epoch 420, loss: 0.0048, correct: 50, time per epoch: 0.1261 seconds
Epoch 430, loss: 0.3255, correct: 50, time per epoch: 0.1251 seconds
Epoch 440, loss: 0.0607, correct: 50, time per epoch: 0.1260 seconds
Epoch 450, loss: 0.0098, correct: 50, time per epoch: 0.1361 seconds
Epoch 460, loss: 0.1675, correct: 50, time per epoch: 0.1259 seconds
Epoch 470, loss: 0.1243, correct: 50, time per epoch: 0.1243 seconds
Epoch 480, loss: 0.4404, correct: 50, time per epoch: 0.1257 seconds
Epoch 490, loss: 0.0513, correct: 50, time per epoch: 0.2491 seconds


**Simple GPU:**


![image](https://github.com/user-attachments/assets/51f083c7-174c-4dd5-b4ef-999dad99174b)


![image](https://github.com/user-attachments/assets/cdf54347-4b07-4269-9243-733268b54e0c)


**Split CPU:**

!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05


Epoch 10, loss: 5.0561, correct: 38, time per epoch: 0.1302 seconds
Epoch 20, loss: 4.2774, correct: 34, time per epoch: 0.1781 seconds
Epoch 30, loss: 4.1055, correct: 42, time per epoch: 0.1257 seconds
Epoch 40, loss: 2.9834, correct: 49, time per epoch: 0.1364 seconds
Epoch 50, loss: 2.9245, correct: 49, time per epoch: 0.1260 seconds
Epoch 60, loss: 1.8494, correct: 49, time per epoch: 0.1259 seconds
Epoch 70, loss: 1.8770, correct: 44, time per epoch: 0.1287 seconds
Epoch 80, loss: 2.4550, correct: 49, time per epoch: 0.1249 seconds
Epoch 90, loss: 2.5274, correct: 45, time per epoch: 0.1381 seconds
Epoch 100, loss: 1.9774, correct: 50, time per epoch: 0.1286 seconds
Epoch 110, loss: 1.3875, correct: 50, time per epoch: 0.2750 seconds
Epoch 120, loss: 1.4781, correct: 50, time per epoch: 0.1260 seconds
Epoch 130, loss: 0.8857, correct: 50, time per epoch: 0.1298 seconds
Epoch 140, loss: 0.7940, correct: 50, time per epoch: 0.1450 seconds
Epoch 150, loss: 0.9902, correct: 50, time per epoch: 0.1259 seconds
Epoch 160, loss: 0.4616, correct: 50, time per epoch: 0.1237 seconds
Epoch 170, loss: 1.4463, correct: 50, time per epoch: 0.1241 seconds
Epoch 180, loss: 1.1920, correct: 50, time per epoch: 0.1282 seconds
Epoch 190, loss: 0.6145, correct: 50, time per epoch: 0.1384 seconds
Epoch 200, loss: 0.4867, correct: 50, time per epoch: 0.1424 seconds
Epoch 210, loss: 0.4325, correct: 50, time per epoch: 0.2243 seconds
Epoch 220, loss: 0.6001, correct: 50, time per epoch: 0.1261 seconds
Epoch 230, loss: 0.6966, correct: 50, time per epoch: 0.1239 seconds
Epoch 240, loss: 0.8673, correct: 50, time per epoch: 0.1253 seconds
Epoch 250, loss: 0.8806, correct: 50, time per epoch: 0.1262 seconds
Epoch 260, loss: 0.7801, correct: 49, time per epoch: 0.1282 seconds
Epoch 270, loss: 0.8815, correct: 50, time per epoch: 0.1253 seconds
Epoch 280, loss: 0.2180, correct: 50, time per epoch: 0.1299 seconds
Epoch 290, loss: 0.8105, correct: 50, time per epoch: 0.1257 seconds
Epoch 300, loss: 0.4848, correct: 50, time per epoch: 0.1463 seconds
Epoch 310, loss: 0.6197, correct: 50, time per epoch: 0.1263 seconds
Epoch 320, loss: 0.1401, correct: 50, time per epoch: 0.1263 seconds
Epoch 330, loss: 0.0265, correct: 50, time per epoch: 0.1243 seconds
Epoch 340, loss: 0.6629, correct: 50, time per epoch: 0.1269 seconds
Epoch 350, loss: 0.5603, correct: 50, time per epoch: 0.1416 seconds
Epoch 360, loss: 0.7846, correct: 50, time per epoch: 0.1318 seconds
Epoch 370, loss: 0.2379, correct: 50, time per epoch: 0.1268 seconds
Epoch 380, loss: 0.6424, correct: 50, time per epoch: 0.1259 seconds
Epoch 390, loss: 0.0126, correct: 50, time per epoch: 0.2099 seconds
Epoch 400, loss: 0.1680, correct: 50, time per epoch: 0.2844 seconds
Epoch 410, loss: 0.0879, correct: 50, time per epoch: 0.1277 seconds
Epoch 420, loss: 0.2850, correct: 50, time per epoch: 0.1275 seconds
Epoch 430, loss: 0.2588, correct: 50, time per epoch: 0.1239 seconds
Epoch 440, loss: 0.5862, correct: 50, time per epoch: 0.1281 seconds
Epoch 450, loss: 0.7373, correct: 50, time per epoch: 0.1251 seconds
Epoch 460, loss: 0.1739, correct: 50, time per epoch: 0.1247 seconds
Epoch 470, loss: 0.1141, correct: 50, time per epoch: 0.1256 seconds
Epoch 480, loss: 0.0786, correct: 50, time per epoch: 0.1252 seconds
Epoch 490, loss: 0.1700, correct: 50, time per epoch: 0.1845 seconds


**Split GPU:**


![image](https://github.com/user-attachments/assets/842df77f-e0bb-4614-a9ae-19b4c5ccb885)


![image](https://github.com/user-attachments/assets/ed8949ba-e21c-4b49-b516-548ed410e559)


**Xor CPU:**

!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05


Epoch 10, loss: 6.0262, correct: 29, time per epoch: 0.1950 seconds
Epoch 20, loss: 3.7967, correct: 47, time per epoch: 0.2942 seconds
Epoch 30, loss: 4.1765, correct: 47, time per epoch: 0.1274 seconds
Epoch 40, loss: 2.8464, correct: 47, time per epoch: 0.1396 seconds
Epoch 50, loss: 1.8168, correct: 48, time per epoch: 0.1288 seconds
Epoch 60, loss: 3.4651, correct: 47, time per epoch: 0.1287 seconds
Epoch 70, loss: 2.4421, correct: 45, time per epoch: 0.1276 seconds
Epoch 80, loss: 1.9078, correct: 48, time per epoch: 0.1393 seconds
Epoch 90, loss: 1.8686, correct: 48, time per epoch: 0.1252 seconds
Epoch 100, loss: 1.1628, correct: 48, time per epoch: 0.1291 seconds
Epoch 110, loss: 1.3511, correct: 48, time per epoch: 0.2151 seconds
Epoch 120, loss: 1.4243, correct: 48, time per epoch: 0.1274 seconds
Epoch 130, loss: 0.8387, correct: 48, time per epoch: 0.1372 seconds
Epoch 140, loss: 2.4665, correct: 48, time per epoch: 0.1278 seconds
Epoch 150, loss: 1.9236, correct: 48, time per epoch: 0.1291 seconds
Epoch 160, loss: 1.3520, correct: 48, time per epoch: 0.1312 seconds
Epoch 170, loss: 1.7257, correct: 49, time per epoch: 0.1268 seconds
Epoch 180, loss: 0.3813, correct: 48, time per epoch: 0.1396 seconds
Epoch 190, loss: 0.4178, correct: 48, time per epoch: 0.1257 seconds
Epoch 200, loss: 0.9362, correct: 48, time per epoch: 0.2727 seconds
Epoch 210, loss: 0.3049, correct: 48, time per epoch: 0.1255 seconds
Epoch 220, loss: 1.2961, correct: 50, time per epoch: 0.1266 seconds
Epoch 230, loss: 0.2197, correct: 49, time per epoch: 0.1256 seconds
Epoch 240, loss: 1.0391, correct: 48, time per epoch: 0.1262 seconds
Epoch 250, loss: 0.8525, correct: 50, time per epoch: 0.1265 seconds
Epoch 260, loss: 1.2462, correct: 50, time per epoch: 0.1281 seconds
Epoch 270, loss: 0.4241, correct: 50, time per epoch: 0.1246 seconds
Epoch 280, loss: 2.2656, correct: 48, time per epoch: 0.1334 seconds
Epoch 290, loss: 0.7382, correct: 50, time per epoch: 0.2140 seconds
Epoch 300, loss: 0.3651, correct: 48, time per epoch: 0.1968 seconds
Epoch 310, loss: 1.3865, correct: 50, time per epoch: 0.1262 seconds
Epoch 320, loss: 0.2471, correct: 48, time per epoch: 0.1275 seconds
Epoch 330, loss: 0.8769, correct: 50, time per epoch: 0.1352 seconds
Epoch 340, loss: 0.3867, correct: 50, time per epoch: 0.1260 seconds
Epoch 350, loss: 0.1782, correct: 50, time per epoch: 0.1281 seconds
Epoch 360, loss: 0.9499, correct: 49, time per epoch: 0.1272 seconds
Epoch 370, loss: 0.1555, correct: 50, time per epoch: 0.1285 seconds
Epoch 380, loss: 1.1151, correct: 48, time per epoch: 0.1406 seconds
Epoch 390, loss: 0.3260, correct: 50, time per epoch: 0.2213 seconds
Epoch 400, loss: 0.7980, correct: 50, time per epoch: 0.1260 seconds
Epoch 410, loss: 0.7110, correct: 50, time per epoch: 0.1264 seconds
Epoch 420, loss: 0.3218, correct: 50, time per epoch: 0.1241 seconds
Epoch 430, loss: 0.7746, correct: 50, time per epoch: 0.1287 seconds
Epoch 440, loss: 1.6752, correct: 49, time per epoch: 0.1245 seconds
Epoch 450, loss: 0.6630, correct: 50, time per epoch: 0.1299 seconds
Epoch 460, loss: 0.6586, correct: 50, time per epoch: 0.1271 seconds
Epoch 470, loss: 0.6264, correct: 50, time per epoch: 0.1263 seconds
Epoch 480, loss: 1.0763, correct: 49, time per epoch: 0.2182 seconds
Epoch 490, loss: 0.7174, correct: 50, time per epoch: 0.1300 seconds


**Xor GPU:**


![image](https://github.com/user-attachments/assets/ce778762-5712-44fd-b8fd-5feb9a2652b8)


![image](https://github.com/user-attachments/assets/2142d1fb-b18a-4e26-a6e4-af46717724d8)


**Large Simple CPU (Hidden size = 200):**

!cd $DIR; time PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET simple --RATE 0.05

Epoch 10, loss: 1.4310, correct: 50, time per epoch: 0.2755 seconds
Epoch 20, loss: 0.1680, correct: 49, time per epoch: 0.2779 seconds
Epoch 30, loss: 0.9681, correct: 50, time per epoch: 0.2885 seconds
Epoch 40, loss: 0.7233, correct: 50, time per epoch: 0.2837 seconds
Epoch 50, loss: 1.2803, correct: 50, time per epoch: 0.2795 seconds
Epoch 60, loss: 0.6762, correct: 50, time per epoch: 0.2917 seconds
Epoch 70, loss: 0.9660, correct: 50, time per epoch: 0.2803 seconds
Epoch 80, loss: 1.1535, correct: 50, time per epoch: 0.2789 seconds
Epoch 90, loss: 0.5075, correct: 50, time per epoch: 0.3713 seconds
Epoch 100, loss: 0.4745, correct: 50, time per epoch: 0.3126 seconds
Epoch 110, loss: 0.2760, correct: 50, time per epoch: 0.2768 seconds
Epoch 120, loss: 0.3188, correct: 50, time per epoch: 0.2875 seconds
Epoch 130, loss: 0.9035, correct: 50, time per epoch: 0.4957 seconds
Epoch 140, loss: 0.3957, correct: 50, time per epoch: 0.2777 seconds
Epoch 150, loss: 0.6403, correct: 50, time per epoch: 0.2922 seconds
Epoch 160, loss: 0.1785, correct: 50, time per epoch: 0.2827 seconds
Epoch 170, loss: 0.0649, correct: 50, time per epoch: 0.5313 seconds
Epoch 180, loss: 0.5338, correct: 50, time per epoch: 0.2897 seconds
Epoch 190, loss: 0.6457, correct: 50, time per epoch: 0.2755 seconds
Epoch 200, loss: 0.0039, correct: 50, time per epoch: 0.2839 seconds
Epoch 210, loss: 0.4734, correct: 50, time per epoch: 0.2922 seconds
Epoch 220, loss: 0.1688, correct: 50, time per epoch: 0.2760 seconds
Epoch 230, loss: 0.3097, correct: 50, time per epoch: 0.2884 seconds
Epoch 240, loss: 0.6218, correct: 50, time per epoch: 0.2905 seconds
Epoch 250, loss: 0.0347, correct: 50, time per epoch: 0.2883 seconds
Epoch 260, loss: 0.4473, correct: 50, time per epoch: 0.2998 seconds
Epoch 270, loss: 0.4621, correct: 50, time per epoch: 0.2797 seconds
Epoch 280, loss: 0.0331, correct: 50, time per epoch: 0.2816 seconds
Epoch 290, loss: 0.3031, correct: 50, time per epoch: 0.2860 seconds
Epoch 300, loss: 0.0339, correct: 50, time per epoch: 0.5928 seconds
Epoch 310, loss: 0.2371, correct: 50, time per epoch: 0.2790 seconds
Epoch 320, loss: 0.2748, correct: 50, time per epoch: 0.2905 seconds
Epoch 330, loss: 0.2805, correct: 50, time per epoch: 0.2773 seconds
Epoch 340, loss: 0.0759, correct: 50, time per epoch: 0.5611 seconds
Epoch 350, loss: 0.1209, correct: 50, time per epoch: 0.2872 seconds
Epoch 360, loss: 0.1662, correct: 50, time per epoch: 0.2794 seconds
Epoch 370, loss: 0.1313, correct: 50, time per epoch: 0.2743 seconds
Epoch 380, loss: 0.0569, correct: 50, time per epoch: 0.3660 seconds
Epoch 390, loss: 0.2029, correct: 50, time per epoch: 0.2752 seconds
Epoch 400, loss: 0.2264, correct: 50, time per epoch: 0.2916 seconds
Epoch 410, loss: -0.0000, correct: 50, time per epoch: 0.2745 seconds
Epoch 420, loss: 0.0414, correct: 50, time per epoch: 0.2787 seconds
Epoch 430, loss: 0.0127, correct: 50, time per epoch: 0.2870 seconds
Epoch 440, loss: 0.2886, correct: 50, time per epoch: 0.2808 seconds
Epoch 450, loss: 0.2231, correct: 50, time per epoch: 0.2782 seconds
Epoch 460, loss: 0.0094, correct: 50, time per epoch: 0.2870 seconds
Epoch 470, loss: 0.0340, correct: 50, time per epoch: 0.2772 seconds
Epoch 480, loss: 0.0210, correct: 50, time per epoch: 0.2950 seconds
Epoch 490, loss: 0.0287, correct: 50, time per epoch: 0.2853 seconds


**Large Simple GPU (Hidden size = 200):**


![image](https://github.com/user-attachments/assets/9c204efd-c8cf-434b-ad15-ffd228d19f74)


![image](https://github.com/user-attachments/assets/e8062c6b-7177-4a66-a0cf-c5aba3f56064)
