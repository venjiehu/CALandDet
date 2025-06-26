import os
import glob
import torch
import pickle
import torchvision
import numpy as np
from datetime import datetime
from PIL import Image
from osgeo import gdal, gdalconst
from pathlib import Path
from einops import rearrange
from collections import OrderedDict
import multiprocessing as mp
from multiprocessing import Process, Manager


def _correct_coord(w_j, patch_size, img_w, sample_size):
    """
    :param w_j:当前x的坐标，
    :param patch_size: 一个块的大小
    :param img_w: 整个x方向的宽度
    :param sample_size: 单个样本大小
    :return: x的坐标，x方向的patch宽度，以及多出了多少pix
    """
    x_w_off = w_j
    x_w_size = patch_size
    x_gap = 0  # x方向不够整sample的像素值

    # 判断x方向是否可以填满patch，当不足一个patch是修改参数使其能正好满足样本尺寸
    if w_j + patch_size > img_w:
        gap_j_pix = img_w - w_j
        gap_j_n = gap_j_pix // sample_size  # 需要多少块
        x_r = gap_j_pix % sample_size  # 余数，该值不为0，表明存在多余不满一个样本尺寸的像素值
        gap_j_n = gap_j_n if x_r == 0 else gap_j_n + 1  # 补上像素，使其尺寸满足样本大小
        x_w_size = gap_j_n * sample_size
        x_gap = sample_size - x_r
        x_w_off = x_w_off - x_gap

    return x_w_off, x_w_size, x_gap


def _save_png(path, file_name, data, bands):
    data = data.transpose(1, 2, 0)
    if bands == 4:
        img = Image.fromarray(data, mode='RGBA').convert("RGB")
    elif bands == 3:
        img = Image.fromarray(data, mode='RGB')
    else:
        raise ValueError("输入数据仅支持RBG或者RGBA通道的数据！！！")

    img.save(os.path.join(path, file_name))


def tif2patch(path, sample_size, patch_size, overlap_rate, work_path, work_symbol, lock, patch_list):
    """
     :param path: 输入tif数据的具体位置
     :param sample_size: 每个样本的长宽尺寸，只考虑样本为矩形
     :param patch_size: 每个batch块包含的patch尺寸
     :param overlap_rate: 样本重叠率
     :param work_path: 工作文件夹
     :param patch_list:存在用于进程共享的patch名字
     :param work_symbol:用于只是该模块是否在工作
     :param lock:用于进程锁
     :return:
     """

    dataset = gdal.Open(path)
    img_w = dataset.RasterXSize  # 栅格矩阵的列数
    img_h = dataset.RasterYSize  # 栅格矩阵的行数
    img_bands = dataset.RasterCount  # 波段数
    img_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
    img_proj = dataset.GetProjection()  # 获取投影信息
    img_info = OrderedDict()
    img_info.update({
        "h": img_h,
        "w": img_w,
        "bands": img_bands,
        "img_geotrans": img_geotrans,
        "img_proj": img_proj
    })
    #
    with open(os.path.join(work_path, "img_info.pkl"), 'wb') as f:
        pickle.dump(img_info, f)
    # 创建 img2patch保存文件夹
    save_path = os.path.join(work_path, "img2patch")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #
    overlap = int(sample_size * overlap_rate)
    # 定义初始工作位置
    h_i = 0

    while h_i < img_h:
        w_j = 0  # 初始化

        # 初始化y方向
        y_h_off, y_h_size, y_gap = _correct_coord(h_i, patch_size, img_h, sample_size)

        while w_j < img_w:
            x_w_off, x_w_size, x_gap = _correct_coord(w_j, patch_size, img_w, sample_size)
            cut_data = dataset.ReadAsArray(xoff=x_w_off, yoff=y_h_off, xsize=x_w_size, ysize=y_h_size)
            file_name = f"{y_h_off}_{x_w_off}_{y_h_size}_{x_w_size}_{y_gap}_{x_gap}.png"

            # 保存patch图片
            _save_png(save_path, file_name, cut_data, img_bands)
            w_j += (patch_size - overlap)

            # 将文件名字保存在列表中以便其他进程共享
            with lock:
                patch_list.append(file_name)

        h_i += (patch_size - overlap)

    # 完成工作
    with lock:
        work_symbol.value = not work_symbol.value


def _data_pre(img_path, sample_size, overlap_rate, device):
    # 定义初始化函数
    fn_nor = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    unfold = torch.nn.Unfold(kernel_size=(sample_size, sample_size), stride=int(sample_size * overlap_rate))
    # 读取数据
    img_data = Image.open(img_path)

    w, h = img_data.size
    img_data = np.array(img_data)

    # 将 NumPy 数组转换为 PyTorch 张量
    img_data = torch.from_numpy(img_data).to(device)

    # 转换为 PyTorch 默认的 float32 类型，并且改变维度顺序为 (channels, height, width)
    img_data = img_data.permute(2, 0, 1).float()
    # 归一化数据
    img_data = fn_nor(img_data / 255.)
    # 将patch分割为模型需要输入的格式
    output = unfold(img_data)
    output = output.view(3, sample_size, sample_size, -1)
    output = rearrange(output, "c h w n -> n c h w")

    return output, h, w


def _save_result(output, sample_size, h, w, overlap_rate, save_path, name: str):
    # 定义翻转函数
    fold = torch.nn.Fold(output_size=(h, w), kernel_size=(sample_size, sample_size),
                         stride=int(sample_size * overlap_rate))
    re_output = torch.zeros_like(output)
    # 舍弃部分重叠区域
    offset = int((sample_size * overlap_rate) / 2)
    offset_size = offset + int(sample_size * (1 - overlap_rate))
    re_output[:, :, offset:offset_size, offset:offset_size] = output[:, :, offset:offset_size, offset:offset_size]

    re_output = rearrange(re_output, " b c h w-> c h w b")
    re_output = re_output.view(sample_size * sample_size, -1)
    # 将变量转换为 numpy形式
    reconstructed = fold(re_output).cpu().numpy()
    reconstructed = np.array(reconstructed, dtype=np.float16).reshape(h, w)
    #
    # 保存为numpy文件
    name = name.split(".")[0] + ".npz"
    np.savez_compressed(os.path.join(save_path, name), reconstructed)


def seg_work(work_path, model: torch.nn.Module, sample_size, overlap_rate, patch_list, work_symbol, lock,
             device=torch.device("cuda:0"), chunk_nums: int = 1):
    """
    :param work_path: 工作路径
    :param model: 用于推理的模型
    :param sample_size: 模型预测样本的大小
    :param overlap_rate: 重叠率
    :param patch_list: 需要处理的patch列表
    :param work_symbol: 表明tif2patch进程是否处理完毕
    :param lock: 进程锁，用于多线程
    :param device: 选择gpu作为推理设备
    :param chunk_nums: 如果patch太大，超出显存限制，则调整该参数。
    :return:
    """
    # 获取工作路径
    img_work_path = Path(os.path.join(work_path, "img2patch"))
    # 创建存放预测结果的文件夹
    result_path = os.path.join(work_path, "model_inference")
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # 将模型设置为评估模型，并将其移动到gpu
    model.eval()
    model.to(device)

    with torch.no_grad():
        # 如果tif2patch模块没有停止工作，则该函数应该挂起，且列表不为空的时候应该继续工作直到列表为空
        while True:

            with lock:
                if not work_symbol.value and len(patch_list) == 0:
                    break

                if len(patch_list) > 0:
                    name = patch_list.pop()
                else:
                    continue

            img_path = os.path.join(img_work_path, name)
            print(img_path)
            input_data, h, w = _data_pre(img_path, sample_size, overlap_rate, device)

            outputs = []
            input_chunks = torch.chunk(input_data, chunk_nums)
            for input_chunk in input_chunks:
                chunk_output = model(input_chunk).detach().sigmoid()
                outputs.append(chunk_output)

            # 合并结果
            out_put = torch.cat(outputs, dim=0)

            _save_result(out_put, sample_size, h, w, overlap_rate, result_path, name)


def merge_result(work_path, sample_size, overlap_rate, threshold=0.5, flush_cache_interval=100):
    # 获取源样本栅格数据
    img_info_path = os.path.join(work_path, "img_info.pkl")
    with open(img_info_path, 'rb') as f:
        img_info_dict = pickle.load(f)
        img_h = img_info_dict["h"]
        img_w = img_info_dict["w"]
        img_geotrans = img_info_dict["img_geotrans"]
        img_proj = img_info_dict["img_proj"]

    # 获取需要模型预测结果路径列表
    img_npz_paths = glob.glob(os.path.join(work_path, "model_inference", "*.npz"))

    # 创建一个新的 GeoTIFF 文件,波段设置为1
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(os.path.join(work_path, 'inference_mask.tif'), img_w, img_h, 1, gdalconst.GDT_Byte)

    # 设置地理变换参数和投影信息
    dataset.SetGeoTransform(img_geotrans)
    dataset.SetProjection(img_proj)

    # 获取第一个波段（GDAL 中的波段索引从 1 开始，而不是从 0 开始）
    band = dataset.GetRasterBand(1)
    #
    overlap_off_pix = int(sample_size * overlap_rate / 2)
    counter = 0
    for img_npz in img_npz_paths:
        # 读取预测结果
        name = Path(img_npz).name.split(".")[0]
        img_arrays = np.load(img_npz)
        img_arrays = img_arrays['arr_0']

        # 获取相关patch块的位置信息，并将其转换为int类型用于后续计算
        y_h_off, x_w_off, y_h_size, x_w_size, y_gap, x_gap = list(map(int, name.split("_")))
        # 裁掉重复的黑边，考虑到凑整补充的像素
        img_y_off = overlap_off_pix + y_gap
        img_x_off = overlap_off_pix + x_gap
        img_arrays = img_arrays[img_y_off:y_h_size - overlap_off_pix, img_x_off:x_w_size - overlap_off_pix]

        # 修正裁剪之后图片的位置
        y_h_off += (overlap_off_pix - y_gap)
        x_w_off += (overlap_off_pix - x_gap)

        # 修改对应区域的数值，将区域二值化
        img_arrays[img_arrays > threshold] = 1
        img_arrays[img_arrays <= threshold] = 0
        band.WriteArray(img_arrays, xoff=x_w_off, yoff=y_h_off)
        # 计数器加一，当达到刷新的间隔时候，将会执行一次写入磁盘操作避免内存溢出
        counter += 1
        if counter >= flush_cache_interval:
            band.FlushCache()
            counter = 0

    # 完成最后的写入后，关闭数据集
    band.FlushCache()
    dataset.FlushCache()
    dataset = None


def inference_tif2mask(seg_model, work_path, img_path, sample_size, patch_size, overlap_rate):
    # 获取当前时间
    start_time = datetime.now()
    date_time = start_time.strftime("%Y-%m-%d-%H_%M_%S")
    # check path
    work_path = os.path.join(work_path, date_time)
    if not os.path.exists(work_path):
        os.makedirs(work_path)

    # 设置初始的多进程方法
    mp.set_start_method('spawn')
    # 创建多线程
    with Manager() as manager:
        patch_list = manager.list()
        tif2path_work = manager.Value('b', True)
        lock = manager.Lock()

        # 初始化多线程
        p_tif2patch = Process(target=tif2patch,
                              args=(img_path, sample_size, patch_size, overlap_rate, work_path, tif2path_work, lock,
                                    patch_list))

        p_seg_work = Process(target=seg_work,
                             args=(work_path, seg_model, sample_size, overlap_rate, patch_list, tif2path_work, lock))

        p_tif2patch.start()
        p_seg_work.start()

        p_tif2patch.join()
        p_seg_work.join()

    # 开始拼接
    print("start merge!")
    merge_result(work_path, sample_size, overlap_rate)
    end_time = datetime.now()
    print(f"time spent: {end_time - start_time}")
