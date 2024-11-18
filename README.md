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


**Simple CPU:**


Epoch  0  loss  5.987989761401322 correct 46
Epoch  10  loss  1.1339808796255015 correct 48
Epoch  20  loss  0.8882560549482106 correct 50
Epoch  30  loss  0.9704325970792871 correct 50
Epoch  40  loss  0.9785705241233401 correct 50
Epoch  50  loss  0.44869973570367305 correct 50
Epoch  60  loss  0.4464265217587732 correct 50
Epoch  70  loss  0.6875875291059014 correct 50
Epoch  80  loss  0.11782128324731915 correct 50
Epoch  90  loss  0.460367723572118 correct 50
Epoch  100  loss  0.2695396503302061 correct 50
Epoch  110  loss  0.1135505296028791 correct 50
Epoch  120  loss  0.08313819955249277 correct 50
Epoch  130  loss  0.2793940761606952 correct 50
Epoch  140  loss  0.006666289007465303 correct 50
Epoch  150  loss  0.26709657594694425 correct 50
Epoch  160  loss  0.0786826700620051 correct 50
Epoch  170  loss  0.208318904564589 correct 50
Epoch  180  loss  0.14168245548102046 correct 50
Epoch  190  loss  0.3017914865813328 correct 50
Epoch  200  loss  0.08859953236071032 correct 50
Epoch  210  loss  0.21778417478653017 correct 50
Epoch  220  loss  0.0018905388507236224 correct 50
Epoch  230  loss  0.002874191140385747 correct 50
Epoch  240  loss  0.011180092390816214 correct 50
Epoch  250  loss  0.31335670457410425 correct 50
Epoch  260  loss  0.0926941704812431 correct 50
Epoch  270  loss  0.163571133872689 correct 50
Epoch  280  loss  0.17973123611060077 correct 50
Epoch  290  loss  0.09172225899627334 correct 50
Epoch  300  loss  0.09012640045081745 correct 50
Epoch  310  loss  0.017611860061877923 correct 50
Epoch  320  loss  0.10900068888441805 correct 50
Epoch  330  loss  0.015855736485168883 correct 50
Epoch  340  loss  0.1854786829020064 correct 50
Epoch  350  loss  0.017157732016191047 correct 50
Epoch  360  loss  0.15213826902802763 correct 50
Epoch  370  loss  0.014064305092971758 correct 50
Epoch  380  loss  0.06666149732981728 correct 50
Epoch  390  loss  0.06584907659354007 correct 50
Epoch  400  loss  0.1270662809791634 correct 50
Epoch  410  loss  0.12865624231097506 correct 50
Epoch  420  loss  0.0005563460051300604 correct 50
Epoch  430  loss  0.009229141636151784 correct 50
Epoch  440  loss  0.011249221201719435 correct 50
Epoch  450  loss  0.09561928658656552 correct 50
Epoch  460  loss  0.08449537964163131 correct 50
Epoch  470  loss  0.03293552052854228 correct 50
Epoch  480  loss  0.009344572429578195 correct 50
Epoch  490  loss  0.002709744625298445 correct 50

real	1m41.118s
user	1m50.832s
sys	0m21.237s

Time per epoch = 101.118 seconds / 500 epochs = 0.202236s / epoch


**Simple GPU:**


![image](https://github.com/user-attachments/assets/51f083c7-174c-4dd5-b4ef-999dad99174b)


![image](https://github.com/user-attachments/assets/cdf54347-4b07-4269-9243-733268b54e0c)


**Split CPU:**


Epoch  0  loss  7.5443507036846995 correct 33
Epoch  10  loss  4.340884043556672 correct 37
Epoch  20  loss  4.4363487723466415 correct 47
Epoch  30  loss  3.5529081389205635 correct 48
Epoch  40  loss  2.5977617404279068 correct 49
Epoch  50  loss  1.905610976391001 correct 49
Epoch  60  loss  1.495528347771917 correct 49
Epoch  70  loss  2.2333177650638327 correct 50
Epoch  80  loss  0.9427869896901846 correct 49
Epoch  90  loss  1.7547524921269928 correct 50
Epoch  100  loss  0.43255409490116525 correct 49
Epoch  110  loss  0.7023494633224893 correct 50
Epoch  120  loss  1.2139813502515302 correct 50
Epoch  130  loss  0.806659638574395 correct 50
Epoch  140  loss  1.310726586705014 correct 50
Epoch  150  loss  0.3166418238174884 correct 50
Epoch  160  loss  0.786022002804293 correct 50
Epoch  170  loss  0.6722947308370519 correct 50
Epoch  180  loss  0.2154394706368696 correct 50
Epoch  190  loss  0.548708848133395 correct 50
Epoch  200  loss  0.36335557344461294 correct 50
Epoch  210  loss  0.2228308060527873 correct 50
Epoch  220  loss  0.33803861103450905 correct 50
Epoch  230  loss  0.21357286193050945 correct 50
Epoch  240  loss  0.28170166733306456 correct 50
Epoch  250  loss  0.3622322777857479 correct 50
Epoch  260  loss  0.0578161128531061 correct 50
Epoch  270  loss  0.350264307508959 correct 50
Epoch  280  loss  0.16247416377441326 correct 50
Epoch  290  loss  0.16435053953087544 correct 50
Epoch  300  loss  0.20010387926355416 correct 50
Epoch  310  loss  0.35213119662021175 correct 50
Epoch  320  loss  0.25680043490002086 correct 50
Epoch  330  loss  0.23823693832615483 correct 50
Epoch  340  loss  0.15720120371043483 correct 50
Epoch  350  loss  0.08309008338729987 correct 50
Epoch  360  loss  0.136156536373216 correct 50
Epoch  370  loss  0.12326044060767562 correct 50
Epoch  380  loss  0.12278165116859091 correct 50
Epoch  390  loss  0.01933774242321986 correct 50
Epoch  400  loss  0.2744961514610852 correct 50
Epoch  410  loss  0.044869734614659464 correct 50
Epoch  420  loss  0.13656515348210052 correct 50
Epoch  430  loss  0.1141825268115125 correct 50
Epoch  440  loss  0.21231354760410953 correct 50
Epoch  450  loss  0.16948531633631375 correct 50
Epoch  460  loss  0.1960632220749851 correct 50
Epoch  470  loss  0.07804057635753071 correct 50
Epoch  480  loss  0.05403653810224157 correct 50
Epoch  490  loss  0.025156950929326658 correct 50

real	1m42.023s
user	1m50.836s
sys	0m21.178s

Time per epoch = 102.023 seconds / 500 epochs = 0.204046s / epoch

**Split GPU:**


![image](https://github.com/user-attachments/assets/842df77f-e0bb-4614-a9ae-19b4c5ccb885)


![image](https://github.com/user-attachments/assets/ed8949ba-e21c-4b49-b516-548ed410e559)


**Xor CPU:**


Epoch  0  loss  5.991709204708735 correct 32
Epoch  10  loss  5.766563853329427 correct 41
Epoch  20  loss  3.297786250114357 correct 41
Epoch  30  loss  2.220942568453458 correct 43
Epoch  40  loss  3.4561159701634403 correct 45
Epoch  50  loss  3.2705224202005163 correct 45
Epoch  60  loss  3.5626964533810717 correct 46
Epoch  70  loss  2.21373111622788 correct 46
Epoch  80  loss  3.061618856708214 correct 46
Epoch  90  loss  2.499616712182128 correct 47
Epoch  100  loss  1.0125564364894193 correct 47
Epoch  110  loss  1.6636024472065742 correct 49
Epoch  120  loss  1.836710406303256 correct 48
Epoch  130  loss  2.76869061204275 correct 48
Epoch  140  loss  2.969009128944592 correct 47
Epoch  150  loss  1.8470094832055928 correct 48
Epoch  160  loss  1.957274136979041 correct 49
Epoch  170  loss  0.5471992711266982 correct 50
Epoch  180  loss  0.755463813884341 correct 49
Epoch  190  loss  1.2070154680869314 correct 50
Epoch  200  loss  0.7446380970481395 correct 49
Epoch  210  loss  0.9242818071020422 correct 49
Epoch  220  loss  1.2168588010628287 correct 50
Epoch  230  loss  1.1822295302673085 correct 50
Epoch  240  loss  1.8381563751838466 correct 50
Epoch  250  loss  0.5456675863045639 correct 49
Epoch  260  loss  1.0164296974505893 correct 49
Epoch  270  loss  0.6246719893930509 correct 49
Epoch  280  loss  0.635823113776489 correct 50
Epoch  290  loss  1.1422879881734986 correct 50
Epoch  300  loss  0.4420426027426594 correct 50
Epoch  310  loss  0.3222082607068874 correct 50
Epoch  320  loss  0.3468941659694837 correct 49
Epoch  330  loss  0.36436227682086697 correct 50
Epoch  340  loss  0.4173386558576742 correct 49
Epoch  350  loss  1.071934400595146 correct 49
Epoch  360  loss  0.7591724676619556 correct 49
Epoch  370  loss  0.23266170806663322 correct 50
Epoch  380  loss  0.29519695435418375 correct 50
Epoch  390  loss  0.7153003985155963 correct 49
Epoch  400  loss  1.0471972942661596 correct 49
Epoch  410  loss  0.2545563246758732 correct 50
Epoch  420  loss  0.3927770914135405 correct 49
Epoch  430  loss  0.27807129739368863 correct 50
Epoch  440  loss  0.10280581424260334 correct 50
Epoch  450  loss  0.31456399484019837 correct 50
Epoch  460  loss  0.19475750894883465 correct 49
Epoch  470  loss  0.9944357402929329 correct 49
Epoch  480  loss  1.0670927589295072 correct 49
Epoch  490  loss  0.9260404669776923 correct 50

real	1m42.092s
user	1m50.647s
sys	0m21.689s


Time per epoch = 102.092 seconds / 500 epochs = 0.204184s / epoch


**Xor GPU:**


![image](https://github.com/user-attachments/assets/ce778762-5712-44fd-b8fd-5feb9a2652b8)


![image](https://github.com/user-attachments/assets/2142d1fb-b18a-4e26-a6e4-af46717724d8)


**Large Simple CPU (Hidden size = 200):**


Epoch  0  loss  25.74477478629269 correct 30
Epoch  10  loss  2.587440686376291 correct 48
Epoch  20  loss  1.8326007625188308 correct 48
Epoch  30  loss  0.3008866880092097 correct 50
Epoch  40  loss  0.18493383124412272 correct 50
Epoch  50  loss  0.5584817676373124 correct 49
Epoch  60  loss  0.18610453176024153 correct 50
Epoch  70  loss  0.48486699641212216 correct 50
Epoch  80  loss  0.6696157332230048 correct 50
Epoch  90  loss  0.8421543249046683 correct 50
Epoch  100  loss  0.2668294955487722 correct 50
Epoch  110  loss  0.5810589471981162 correct 50
Epoch  120  loss  0.24942955284655763 correct 50
Epoch  130  loss  0.29335969236470494 correct 50
Epoch  140  loss  0.09737390534980594 correct 50
Epoch  150  loss  0.22027489696437919 correct 50
Epoch  160  loss  0.3871980272524356 correct 50
Epoch  170  loss  0.12667166683798817 correct 50
Epoch  180  loss  0.27893026726650183 correct 50
Epoch  190  loss  0.08368197729547652 correct 50
Epoch  200  loss  0.14831692688865267 correct 50
Epoch  210  loss  0.12325288830714544 correct 50
Epoch  220  loss  0.02874224741000093 correct 50
Epoch  230  loss  0.09493439052602573 correct 50
Epoch  240  loss  0.21187218575499078 correct 50
Epoch  250  loss  0.2243361266323303 correct 50
Epoch  260  loss  0.1665639470037431 correct 50
Epoch  270  loss  0.10241223083977077 correct 50
Epoch  280  loss  0.13819909649828924 correct 50
Epoch  290  loss  0.007569198102447756 correct 50
Epoch  300  loss  0.13080052136585285 correct 50
Epoch  310  loss  0.0029800917265263758 correct 50
Epoch  320  loss  0.06816127599047789 correct 50
Epoch  330  loss  0.18017699694391853 correct 50
Epoch  340  loss  0.016597737388972283 correct 50
Epoch  350  loss  0.028177018645851093 correct 50
Epoch  360  loss  0.08010506218411219 correct 50
Epoch  370  loss  0.012230172303616537 correct 50
Epoch  380  loss  0.11385952875109706 correct 50
Epoch  390  loss  0.13687165532641354 correct 50
Epoch  400  loss  0.12156353778078369 correct 50
Epoch  410  loss  0.045150520963701056 correct 50
Epoch  420  loss  0.0323369193796712 correct 50
Epoch  430  loss  0.10019160422281789 correct 50
Epoch  440  loss  0.10972929993777135 correct 50
Epoch  450  loss  0.04045598978374423 correct 50
Epoch  460  loss  0.08103116406914339 correct 50
Epoch  470  loss  0.022267100868911767 correct 50
Epoch  480  loss  0.04720541353818516 correct 50
Epoch  490  loss  0.0827390363276837 correct 50

real	3m2.128s
user	3m37.227s
sys	0m32.672s


Time per epoch = 182.128 seconds / 500 epochs = 0.364256s / epoch


**Large Simple GPU (Hidden size = 200):**


![image](https://github.com/user-attachments/assets/9c204efd-c8cf-434b-ad15-ffd228d19f74)


![image](https://github.com/user-attachments/assets/e8062c6b-7177-4a66-a0cf-c5aba3f56064)


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
