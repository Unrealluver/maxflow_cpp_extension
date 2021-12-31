# CPP重构模块以加速深度学习训练实例

## Abstract

1. 编写一个快速 Maxflow / Mincut 实现, 以减少CPU上运行的时间开销

## Motivation

1.	Deep Graph Cut Network (DGCN) 中需要用到 Maxflow 算法来生成监督信息(generate supervation), 直接使用DGCN算法中的实现需要消耗较多时间, 形成 GPU 大量空闲时间.
2.	通过编写一个快速的实现, 能够减少 CPU 运算耗时, 极大程度压缩每一次 GPU 运算之间的真空期, 提升 GPU 利用率 (GPU Volatile) 的同时, 减少每次训练的时间. 

## Relative Work

1. DGCN
2. GraphCut Algorithm

## Benchmark

1. 采用 DGCN code 中的原始实现: `generate supervision` 的性能指标作为基础的Benchmark, 并使用Pycarm自带的Profile工具进行跑表耗时评测.
2. 紧接着分析 Benchmark 中的性能开销, 集中耗时操作, 由易到难考虑方法实现.
3. Code:
    <details>
    <summary> Click to Show Detailed Code</summary>

        def generate_supervision(feature, label, cues, mask, pred, knn_matrix):
            batchsize, class_num, h, w = pred.shape
            Y = torch.zeros(batchsize, class_num, h, w)
            supervision = cues.clone()

            for i in range(batchsize):
                label_class = torch.nonzero(label[i])
                markers_new = np.zeros((h, w))
                markers_new.fill(NUM_CLASSES)
                pos = np.where(cues[i].numpy() == 1)
                markers_new[pos[1], pos[2]] = pos[0]
                markers_new_flat = markers_new.reshape(h*w)
                for c in (label_class):
                    c_c = c[0].numpy()
                    feature_c = feature[i].reshape(feature.shape[1], h * w).transpose(1, 0)
                    pred_c = pred[i][c[0]]
                    pred_c_flat = pred_c.flatten()
                    g = maxflow.Graph[float]()
                    nodes = g.add_nodes(h * w)
                    pos = np.where(markers_new_flat == c_c)
                    for node_i in pos[0]:
                        g.add_tedge(nodes[node_i], 0, 10)
                        k_neighbor = np.where(knn_matrix[i][node_i] == 1)
                        for neighbor in (k_neighbor[0]):
                            g.add_edge(nodes[node_i], nodes[neighbor], 1, 1)
                    pos = np.where(markers_new_flat == NUM_CLASSES)
                    for node_i in pos[0]:
                        g.add_tedge(nodes[node_i], -np.log10(pred_c_flat[node_i]), -np.log10(1.0 - pred_c_flat[node_i]))
                        k_neighbor = np.where(knn_matrix[i][node_i] == 1)
                        for neighbor in (k_neighbor[0]):
                            g.add_edge(nodes[node_i], nodes[neighbor], 1, 1)
                    pos = np.where((markers_new_flat!= NUM_CLASSES)&(markers_new_flat!=c_c))
                    for node_i in pos[0]:
                        g.add_tedge(nodes[node_i], 10, 0)
                        k_neighbor = np.where(knn_matrix[i][node_i] == 1)
                        for neighbor in (k_neighbor[0]):
                            g.add_edge(nodes[node_i], nodes[neighbor], 1, 1)

                    flow = g.maxflow()
                    node_ids = np.arange(h*w)
                    label_new = g.get_grid_segments(node_ids)

                    # change: debug suervision not same
                    supervision[i][c[0]] = torch.from_numpy(np.where(pred_c>0.7,label_new.astype(int).reshape(h, w),supervision[i][c[0]])).float()

            return supervision

    </details>

4. Benchmark Setting:
    1. `batchsize`为8, `测试显卡`使用TitanXp, 使用`time`模块统计运行40个iter的耗时, 其中`train` 之前的code时间开销为5.05s.
    2. 相关package版本:
       * PyTorch==1.8.1
       * PyMaxflow==1.2.12
       * Numpy==1.20.2
5. Benchmark Profile Result:

   * 总计时间消耗: 67.31s
   * profile 细则:

<div class="center">

| Name                   | Call Count | Time(ms) | Percent | Own Time(ms) | Percent |
|------------------------|------------|----------|---------|--------------|---------|
| generate_supervision   | 40         | 26248    | 35.8%   | 19221        | 26.2%   |
| «method 'cpu'          | 360        | 11687    | 15.9%   | 11687        | 15.9%   |
| «method 'mul_'         | 12324      | 6134     | 8.4%    | 6134         | 8.4%    |
| «method 'add_'         | 24964      | 5464     | 7.5%    | 5464         | 7.5%    |
| «method 'inference'    | 320        | 4044     | 5.5%    | 4044         | 5.5%    |
| «method 'run backward' | 40         | 3565     | 4.9%    | 3565         | 4.9%    |
| «method 'add'          | 12640      | 3529     | 4.8%    | 3529         | 4.8%    |
| «method 'add_edge"     | 12072959   | 2772     | 3.7%    | 2772         | 3.7%    |
| «method 'cuda'         | 706        | 2417     | 3.3%    | 2388         | 3.3%    |

</div>


## Methods

1. `MFF(MaxFlow Forward)`: 考虑到原始的PyMaxflow实现是CPython实现, 目前更多主流方式采取Pybind11实现. 可以先尝试采取C++进行实现, 通过Pybind11封装为.so文件, 再通过setuptools安装为python支持的依赖.
   1. Code:
      1. CPP端:
           <details>
           <summary> Click to Show Detailed Code</summary>
           
               #include <iostream>
               #include <vector>
               #include <torch/extension.h>

               #include <pybind11/pybind11.h>
               #include <pybind11/numpy.h>
               #include <cmath>
               #include "./include/graph.h"

               #include <utility>
               #include <thread>
               #include <chrono>
               #include <functional>
               #include <atomic>
               #include <iterator>

               namespace py = pybind11;

               void maxflow_forward_kernel(
                   float* markers_new_buf_ptr,
                   float* pred_c_buf_ptr,
                   float* knn_matrix_img_buf_ptr,
                   float* result_buf_ptr,
                   int c_c, int i, int h, int w
               ) {
                   Graph<float, float, float> g = Graph<float, float, float>(h * w + 2, h * w * (11 + 2) * 2);  // 43706, k = 10
                   g.add_node(h * w);

                   for(int i = 0; i < h; i++){
                       for (int j = 0; j < w; j++){
                           if(fabs(markers_new_buf_ptr[i * h + j] - c_c) < 1e-6) {
                               g.add_tweights(i * h + j, 0.0, 10.0);
                           }
                           else if (fabs(markers_new_buf_ptr[i * h + j] - 21) < 1e-6) {
                               g.add_tweights(i * h + j, -log10(pred_c_buf_ptr[i * h + j]), -log10(1.0 - pred_c_buf_ptr[i * h + j]));
                           }
                           else {
                               g.add_tweights(i * h + j, 10.0, 0.0);
                           }
                           for(int k = 0; k < h * w; k++) {
                               if(fabs(knn_matrix_img_buf_ptr[(i * h + j) * (h * w) + k] - 1.0) < 1e-6) {
                                   g.add_edge(i * h + j, k, 1.0, 1.0);
                               }
                           }
                       }
                   }
                   float flow = g.maxflow();

                   for(int i = 0; i < h; i++) {
                       for(int j = 0; j < w; j++) {
                           if(pred_c_buf_ptr[i * h + j] > 0.7) {
                               result_buf_ptr[i * h + j] = g.what_segment(i * h + j);
                           }
                           else {
                               if (fabs(markers_new_buf_ptr[i * h + j] - c_c) < 1e-6) {
                                   result_buf_ptr[i * h + j] = 1;
                               }
                               else {
                                   result_buf_ptr[i * h + j] = 0;
                               }
                           }
                       }
                   }
               }



               py::array_t<float> maxflow_forward(
                   py::array_t<float>& markers_new,
                   py::array_t<float>& pred_c,
                   py::array_t<float>& knn_matrix_img,
                   int c_c, int i, bool if_release_gil=false
                   )
               {
                   if (if_release_gil) {
                       py::gil_scoped_release release;     // 释放GIL锁
                   }
                   // initial
                   py::buffer_info markers_new_buf = markers_new.request();
                   py::buffer_info pred_c_buf = pred_c.request();
                   py::buffer_info knn_matrix_img_buf = knn_matrix_img.request();

                   if (markers_new_buf.ndim != 2 || pred_c_buf.ndim != 2 || knn_matrix_img_buf.ndim != 2)
                   {
                       throw std::runtime_error("numpy.ndarray dims must be 2!");
                   }
                   if ((markers_new_buf.shape[0] != pred_c_buf.shape[0])|| (markers_new_buf.shape[1] != pred_c_buf.shape[1]))
                   {
                       throw std::runtime_error("two array shape must be match!");
                   }

                   float* markers_new_buf_ptr = (float*)markers_new_buf.ptr;
                   float* pred_c_buf_ptr = (float*)pred_c_buf.ptr;
                   float* knn_matrix_img_buf_ptr = (float*)knn_matrix_img_buf.ptr;

                   auto result = py::array_t<float>(pred_c_buf.size);
                   //转换为2d矩阵
                   result.resize({pred_c_buf.shape[0],pred_c_buf.shape[1]});
                   py::buffer_info result_buf = result.request();
                   float* result_buf_ptr = (float*)result_buf.ptr;

                   maxflow_forward_kernel(markers_new_buf_ptr, pred_c_buf_ptr, knn_matrix_img_buf_ptr, result_buf_ptr, c_c, i, pred_c_buf.shape[0], pred_c_buf.shape[1]);


                   if (if_release_gil) {
                       py::gil_scoped_acquire acquire;     // C++执行结束前恢复GIL锁
                   }
                   return result;
               }

               PYBIND11_MODULE(maxflow_dgcn_cpp, m) {
                   m.def("forward", &maxflow_forward, "maxflow forward");
               }
           </details>

       2. Setup Tool端:
           <details>
           <summary> Click to Show Detailed Code</summary>

               from setuptools import setup, Extension
               import torch
               from torch.utils import cpp_extension
               import pybind11

               setup(name='maxflow_dgcn_cpp',
                   ext_modules=[cpp_extension.CppExtension(name='maxflow_dgcn_cpp',
                                                           sources=[
                                                                   'maxflow_dgcn.cpp',
                                                           ])],
                   include_dirs=[
                       pybind11.get_include(),
                   ],
                   cmdclass={'build_ext': cpp_extension.BuildExtension}
               )
           </details>
       
       3. Python 调用端: 
           <details>
           <summary> Click to Show Detailed Code</summary>

               import maxflow_dgcn_cpp

               ···

               def generate_supervision_by_so(feature, label, cues, mask, pred, knn_matrix):
                   batchsize, class_num, h, w = pred.shape
                   supervision = cues.clone()

                   for i in range(batchsize):
                       label_class = torch.nonzero(label[i])
                       markers_new = np.zeros((h, w), dtype=np.float32)
                       markers_new.fill(NUM_CLASSES)
                       pos = np.where(cues[i].numpy() == 1)
                       markers_new[pos[1], pos[2]] = pos[0]
                       knn_matrix_img = knn_matrix[i]
                       for c in (label_class):
                           c_c = c[0].numpy()
                           pred_c = pred[i][c_c]

                           supervision[i][c_c] = torch.from_numpy(maxflow_dgcn_cpp.forward(markers_new, pred_c, knn_matrix_img, c_c, i,))

                   return supervision

               ···

           </details>

   2. `MFF` Profile Result:
      * 总计时间消耗: 47.65s
      * profile 细则:
     
   <div class="center">

    | Name                                        | Call Count | Time(ms) | Percent | Own Time(ms) | Percent |
    |---------------------------------------------|------------|----------|---------|--------------|---------|
    | «method 'cpu'                               | 360        | 11663    | 21.5%   | 11663        | 21.5%   |
    | «method 'add_'                              | 24964      | 7663     | 14.1%   | 7663         | 14.1%   |
    | «built-in method maxflow_dgcn_cpp.forward>  | 798        | 6609     | 12.2%   | 6609         | 12.2%   |
    | «method 'mul_'                              | 12324      | 4897     | 9.0%    | 4897         | 9.0%    |
    | «method 'inference'                         | 320        | 3919     | 7.2%    | 3919         | 7.2%    |
    | «method 'run backward'                      | 40         | 3210     | 5.9%    | 3210         | 5.9%    |
    | «method 'add'                               | 12640      | 2888     | 5.3%    | 2888         | 5.3%    |
    | «method 'cuda'                              | 706        | 2755     | 5.1%    | 2704         | 5.0%    |

   </div>

2. `MFFMT-Py(MaxFlow Forward Multi-Threads by Python)`: 从MFFv1的profile结果中, 我们不难看出, 通过CPP实现的版本在运行速度上有了很大的提升, 从Benchmark版本的耗时26248ms成功缩短到了6609ms, 这时我们尝试使用Python的多线程服务, 将`if_release_gil`的option设置为True, 即CPP依赖库会在运行过程中释放Python的全局GIL锁, 并在运行结束后重新进行获取.
   1. Code:
      1. Python调用端

           <details>
           <summary> Click to Show Detailed Code</summary>

                import threading
                import time

                def generate_supervision_multi_threads(feature, label, cues, mask, pred, knn_matrix):
                    batchsize, class_num, h, w = pred.shape
                    supervision = cues.clone()

                    def call_cpp_extension(lon1, lat1, lon2, lat2, test_cnt):
                        # res = None
                        threadLock.acquire()
                        supervision[test_cnt][lat2] = torch.from_numpy(maxflow_dgcn_cpp.forward(lon1, lat1, lon2, lat2, test_cnt))
                        threadLock.release()
                        # print(res)

                    threadLock = threading.Lock()
                    threads = []

                    threadID = 0
                    for i in range(batchsize):
                        label_class = torch.nonzero(label[i])
                        markers_new = np.zeros((h, w), dtype=np.float32)
                        markers_new.fill(NUM_CLASSES)
                        pos = np.where(cues[i].numpy() == 1)
                        markers_new[pos[1], pos[2]] = pos[0]
                        knn_matrix_img = knn_matrix[i]
                        for c in (label_class):
                            c_c = c[0].numpy()
                            pred_c = pred[i][c_c]
                            # multi threads
                            # t = GraphCutThread(threadID, markers_new, pred_c, knn_matrix_img, c_c, i)
                            t = threading.Thread(target=call_cpp_extension,
                                args=(markers_new, pred_c, knn_matrix_img, c_c, i,))
                            # t.start()
                            threads.append(t)
                            threadID = threadID + 1

                    for t in threads:
                        t.setDaemon(True)
                        t.start()

                    # 等待所有线程完成
                    for t in threads:
                        t.join()
                    print("退出主线程")

                    return supervision
        
           </details>

   2. `MFFMT-Py` Profile Result:
      * 总计时间消耗: 47.87s
      * profile 细则:
    
    <div class='center'>


    | Name                     | Call Count | Time(ms) | Percent | Own Time(ms) | Percent |
    |--------------------------|------------|----------|---------|--------------|---------|
    | «method 'cpu'            | 360        | 11321    | 21.0%   | 11321        | 21.0%   |
    | «method 'acquire'        | 4204       | 7991     | 14.9%   | 7991         | 14.9%   |
    | «method 'mul_'           | 12324      | 6155     | 11.4%   | 6155         | 11.4%   |
    | «method 'add_'           | 24964      | 5643     | 10.5%   | 5643         | 10.5%   |
    | «method 'inference'      | 320        | 3792     | 7.0%    | 3792         | 7.0%    |
    | «method 'add'            | 12640      | 2888     | 5.3%    | 2888         | 5.3%    |
    | «method 'run backward'   | 40         | 3195     | 5.9%    | 3195         | 5.9%    |
    | «method 'cuda'           | 706        | 2471     | 4.6%    | 2440         | 4.5%    |
    | «built-in method conv2d> | 4200       | 1572     | 2.9%    | 1572         | 2.9%    |


    </div>

3. `MFFv2(MaxFlow Forward version 2)`: 针对单线程的`MFF`方法, 我们可以进行code上的重构, 减少不必要的语句, 优化代码的执行逻辑, 即得到`MFFv2`.
   1. Code:
      1. Python调用端

           <details>
           <summary> Click to Show Detailed Code</summary>

                import threading
                import time

                ...

                markers_new_batch = torch.ones((batchsize, 41, 41), dtype=torch.long, device=images.device) * NUM_CLASSES
                pos = torch.where(cues == 1)
                markers_new_batch[pos[0], pos[2], pos[3]] = pos[1]
                supervision2 = generate_supervision_by_so_v2((markers_new_batch).float().cpu().numpy(), labels.numpy(), cues.cpu(),
                                                            probs.cpu().detach().numpy(), knn_matrix.cpu().detach().numpy())

                ...

                def generate_supervision_by_so_v2(markers_new_batch, label_class_batch, cues, pred, knn_matrix):
                    batchsize, class_num, h, w = pred.shape
                    supervision = cues

                    for i in range(batchsize):
                        label_class = label_class_batch[i]
                        knn_matrix_img = knn_matrix[i]
                        for c in range(label_class.size):
                            if label_class[c] == 0:
                                continue
                            pred_c = pred[i][c]
                            supervision[i][c] = torch.from_numpy(maxflow_dgcn_cpp.forward(markers_new_batch[i], pred_c, knn_matrix_img, c, i, False))

                    return supervision
        
           </details>

   2. `MFFv2` Profile Result:
      * 总计时间消耗: 44.37s
      * profile 细则:
    
    <div class='center'>


    | Name                                       | Call Count | Time(ms) | Percent | Own Time(ms) | Percent |
    |--------------------------------------------|------------|----------|---------|--------------|---------|
    | «method 'cpu'                              | 320        | 9617     | 19.0%   | 9617         | 19.0%   |
    | «method 'add_'                             | 24964      | 7741     | 15.3%   | 7741         | 15.3%   |
    | «bullt-1n method maxtlow dgcn cpp.torward> | 798        | 5481     | 10.8%   | 5481         | 10.8%   |
    | «method 'mul_'                             | 12324      | 4909     | 9.7%    | 4909         | 9.7%    |
    | «method 'inference'                        | 320        | 4011     | 7.9%    | 4011         | 7.9%    |
    | «method 'run backward'                     | 40         | 3052     | 5.8%    | 3052         | 6.0%    |
    | «method 'add'                              | 12640      | 2944     | 5.8%    | 2944         | 5.8%    |
    | «method 'cuda'                             | 706        | 2482     | 4.9%    | 2482         | 4.9%    |
    | «built-in method conv2d>                   | 4200       | 1760     | 3.5%    | 1760         | 3.5%    |


    </div>

4. `MFFMT-C++(MaxFlow Forward Multi-Threads by C++)`: 沿着多线程的思路进行优化, 我们将`generate supervision`部分全部交由C++侧进行实现, 在C++内进行多线程的调用.
   1. Code:
      1. C++实现端
           <details>
           <summary> Click to Show Detailed Code</summary>

                ...

                py::array_t<float> generate_supervision_multi_threads(
                    py::array_t<float>& markers_new_batch,
                    py::array_t<float>& label_class_batch,
                    py::array_t<float>& cues,
                    py::array_t<float>& pred,
                    py::array_t<float>& knn_matrix
                    )
                {
                    py::buffer_info markers_new_batch_buf = markers_new_batch.request();
                    py::buffer_info label_class_batch_buf = label_class_batch.request();
                    py::buffer_info cues_buf = cues.request();
                    py::buffer_info pred_buf = pred.request();
                    py::buffer_info knn_matrix_buf = knn_matrix.request();

                    float* markers_new_batch_buf_ptr = (float*)markers_new_batch_buf.ptr;
                    float* label_class_batch_buf_ptr = (float*)label_class_batch_buf.ptr;
                    float* cues_buf_ptr = (float*)cues_buf.ptr;
                    float* pred_buf_ptr = (float*)pred_buf.ptr;
                    float* knn_matrix_buf_ptr = (float*)knn_matrix_buf.ptr;

                    int batchsize = pred_buf.shape[0];
                    int class_num = pred_buf.shape[1];
                    int h = pred_buf.shape[2];
                    int w = pred_buf.shape[3];

                    std::vector<std::thread> threads_pool;

                    for (int i = 0; i < batchsize; i++) {
                        float* label_class_ptr = label_class_batch_buf_ptr + i * class_num;
                        float* knn_matrix_img_ptr = knn_matrix_buf_ptr + i * (h * w) * (h * w);
                        float* markers_new_ptr = markers_new_batch_buf_ptr + i * (h * w);
                        for (int c = 0; c < class_num; c++) {
                            if (fabs(label_class_ptr[c] - 0) < 1e-6) {
                                continue;
                            }
                            float* pred_c_ptr = pred_buf_ptr + i * (class_num * h * w) + c * (h * w);
                            float* result_buf_ptr_each = cues_buf_ptr + i * (class_num * h * w) + c * (h * w);

                            threads_pool.push_back(std::thread(maxflow_forward_kernel, markers_new_ptr, pred_c_ptr, knn_matrix_img_ptr, result_buf_ptr_each, c, i, h, w));
                        }
                    }

                    std::vector<std::thread>::iterator it;//声明一个迭代器，来访问vector容器，作用：遍历或者指向vector容器的元素
                    for (it = threads_pool.begin(); it != threads_pool.end(); it++) {
                        (*it).join();
                    }

                    return cues;
                }

                PYBIND11_MODULE(maxflow_dgcn_cpp, m) {
                    m.def("forward", &maxflow_forward, "maxflow forward");
                    m.def("generate_supervision", &generate_supervision, "generate supervision");
                    m.def("generate_supervision_multi_threads", &generate_supervision_multi_threads, "generate supervision multi threads");
                }
        
           </details>

      2. Python调用端

           <details>
           <summary> Click to Show Detailed Code</summary>

                ...

                markers_new_batch = torch.ones((batchsize, 41, 41), dtype=torch.long, device=images.device) * NUM_CLASSES
                pos = torch.where(cues == 1)
                markers_new_batch[pos[0], pos[2], pos[3]] = pos[1]

                supervision = torch.from_numpy(maxflow_dgcn_cpp.generate_supervision((markers_new_batch).float().cpu().numpy(), labels.numpy(), cues.cpu(),
                                                            probs.cpu().detach().numpy(), knn_matrix.cpu().detach().numpy()))

                ...
        
           </details>

   2. `MFFMT-C++` Profile Result:
      * 总计时间消耗: 42.78s
      * profile 细则:
    
    <div class='center'>


    | Name                                                                   | Call Count | Time(ms) | Percent | Own Time(ms) | Percent |
    |------------------------------------------------------------------------|------------|----------|---------|--------------|---------|
    | «method 'cpu'                                                          | 320        | 9904     | 19.8%   | 9904         | 19.8%   |
    | «method 'mul_'                                                         | 12324      | 6072     | 12.1%   | 6072         | 12.1%   |
    | «method 'add_'                                                         | 24964      | 5609     | 11.2%   | 5609         | 11.2%   |
    | «method 'inference'                                                    | 320        | 4246     | 8.5%    | 4246         | 8.5%    |
    | «method 'add'                                                          | 12640      | 3568     | 7.1%    | 2944         | 5.8%    |
    | «method 'cuda'                                                         | 706        | 3330     | 6.7%    | 3290         | 6.6%    |
    | «method 'run backward'                                                 | 40         | 2974     | 5.9%    | 2974         | 5.9%    |
    | «built-in method maxflow_dgcn_cpp.generate_ supervision_multi_threads> | 40         | 2729     | 5.5%    | 2728         | 5.5%    |
    | «built-in method conv2d>                                               | 4200       | 1689     | 3.4%    | 1689         | 3.4%    |


    </div>

5. `MFFMT-C++v2(MaxFlow Forward Multi-Threads by C++ version 2)`: 同样我们针对前一方法, 进行code上的重构, 尽量精简代码, 消去一些cost比较大的`.cpu()`方法.
   1. `MFFMT-C++` Profile Result:
      * 总计时间消耗: 41.52s
    

## Experiments

<div class='center'>


| Methods     | Prepare Time Cost(s) | 40-iter Total Time Consume(s) | 40-iter Train Time Cost(s) | Train Time-Saving Ratio | Maxflow Forward Time Cost(ms) | Maxflow Forward Time-Saving Ratio |
|-------------|----------------------|-------------------------------|----------------------------|-------------------------|-------------------------------|-----------------------------------|
| Baseline    | 5.05                 | 67.31                         | 62.26                      |                         | 26248                         |                                   |
| MFF         | 5.05                 | 47.65                         | 42.60                      | -31.57%                 | 6609                          | -74.82%                           |
| MFFMT-Py    | 5.05                 | 47.87                         | 42.82                      | -31.22%                 | 7991                          | -69.55%                           |
| MFFv2       | 5.05                 | 44.37                         | 39.32                      | -36.84%                 | 5481                          | -79.11%                           |
| MFFMT-C++   | 5.05                 | 42.78                         | 37.73                      | -39.39%                 | 2729                          | -89.60%                           |
| MFFMT-C++v2 | 5.05                 | 41.52                         | 36.47                      | -41.42%                 | 2729                          | -89.60%                           |
</div>

## Setup

1.	基于C++ API实现定制化Maxflow算子

    代码位置：`./maxflow_cpp_extension`

    运行命令：
    ```
    cd maxflow_cpp_extension
    python setup install --user
    ```

## Reference

* EXTENDING PYTORCH: https://pytorch.org/docs/master/notes/extending.html
* PyDenseCRF: https://github.com/lucasb-eyer/pydensecrf/tree/master/pydensecrf
* Graph Cuts Official Implement: https://github.com/pmneila/graphCuts
* PyBind11: https://pybind11.readthedocs.io/en/stable/index.html
* Python C++ interface >> Numpy: https://pybind11.readthedocs.io/en/latest/advanced/pycpp/numpy.html?highlight=array_t#
* C++11 FAQ: https://wizardforcel.gitbooks.io/cpp-11-faq/content/77.html
* Tabels Generator: https://www.tablesgenerator.com/markdown_tables#
* pybind11 落地实践: https://zhuanlan.zhihu.com/p/444805518
* pybind11—python numpy与C++数据传递: https://www.jianshu.com/p/c912a0a59af9