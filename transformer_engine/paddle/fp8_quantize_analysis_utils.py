from collections import defaultdict
import paddle
import numpy as np


from .recompute import is_in_recompute_phase


class FP8QuantizeAnalysisHelper:
    def __init__(self):
        self.step = 0
        self.state = defaultdict(list)
        self.pname_2_structure_name = {}
        self.enable_analysis = False

    def enable(self):
        self.enable_analysis = True

    def disable(self):
        self.enable_analysis = False

    def is_enable(self):
        return self.enable_analysis

    def set_step(self, step):
        self.step = step

    def set_structure_name_mapping(self, name_mapping):
        # Y = X * W
        # DX = DY * W
        # DW = DY * X

        # X, W, DY
        for pname, structname in name_mapping.items():
            self.pname_2_structure_name[pname] = structname
            self.pname_2_structure_name[pname + "#w"] = structname + "#w"
            self.pname_2_structure_name[pname + "#x"] = structname + "#x"
            self.pname_2_structure_name[pname + "#dy"] = structname + "#dy"

    def get_structure_name(self, pname):
        if pname in self.pname_2_structure_name:
            return self.pname_2_structure_name[pname]
        return ""

    @paddle.no_grad()
    def add_tensor(
        self,
        input_tensor,
        fp8_tensor=None,
        micro_step=0,
        layer_idx=0,
        pname="",
        suffix="",
        fp8_dtype="float8_e4m3fn",
        amax=None,
        scale=None,
    ):
        if suffix:
            pname = pname + "#" + suffix
        analysis_state = {}
        # 如果没有传入amax则使用Current Scaling策略计算当前tensor的amax
        if amax is None:
            amax = paddle.max(paddle.abs(input_tensor)).item()

        # 定义FP8 E4M3的参数
        if fp8_dtype in [6, "float8_e4m3fn"]:
            FP8_MAX = 448.0
            FP8_MIN_NORMALIZED = 2**-10  # E4M3最小正规化数是 2 ** -9, 需要往右继续移动一位
            fp8_dtype = paddle.float8_e4m3fn
        elif fp8_dtype in [7, "float8_e5m2"]:
            FP8_MAX = 57344.0
            FP8_MIN_NORMALIZED = 2**-17  # E4M3最小正规化数是 2 ** -16, 需要往右继续移动一位
            fp8_dtype = paddle.float8_e5m2
        else:
            raise ValueError(f"fp8_dtype {fp8_dtype} error, not supported!")

        # 计算缩放因子，将amax映射到FP8的最大值
        if scale is None:
            scale = FP8_MAX / amax

        # 缩放输入
        quantized = input_tensor * scale

        # 统计下溢为0的元素比例
        # 由于是为了找下溢为0的数量，所以我们找小于最小正规化数的数量
        bf16_zero_count = paddle.sum(quantized == 0).item()
        underflow_mask = (paddle.abs(quantized) <= FP8_MIN_NORMALIZED) & (quantized != 0)
        underflow_count = underflow_mask.sum().item()
        total_count = input_tensor.numel()

        # 计算下溢比例
        underflow_ratio = (underflow_count / total_count).item()

        if fp8_tensor is None:
            fp8_tensor = quantized.to(fp8_dtype)
        elif fp8_tensor.dtype == paddle.uint8:
            fp8_tensor = paddle.view(fp8_tensor, fp8_dtype)

        # 反量化,计算量化误差
        dequantize_tensor = fp8_tensor.to(input_tensor.dtype) / scale
        rmse = ((input_tensor - dequantize_tensor) ** 2).mean().sqrt().item()

        # 计算最大值
        max = paddle.max(input_tensor).item()
        # 计算最大值
        min = paddle.min(input_tensor).item()
        # 计算均值
        mean = paddle.mean(input_tensor)
        # 计算标准差
        std = paddle.std(input_tensor)
        # 计算第四阶中心矩
        fourth_moment = paddle.mean((input_tensor - mean) ** 4)
        # 计算峭度
        if std == 0:
            kurtosis = 0.0
        else:
            kurtosis = (fourth_moment / (std**4)).item()

        # analysis_state['quantized'] = quantized
        # analysis_state['fp8_tensor'] = fp8_tensor
        analysis_state["underflow_ratio"] = underflow_ratio
        analysis_state["rmse"] = rmse
        analysis_state["kurtosis"] = kurtosis
        analysis_state["amax"] = amax
        analysis_state["max"] = max
        analysis_state["min"] = min
        analysis_state["mean"] = mean.item()
        analysis_state["std"] = std.item()
        analysis_state["step"] = self.step
        analysis_state["layer_idx"] = layer_idx
        #analysis_state["pname"] = pname
        #analysis_state["sname"] = self.get_structure_name(pname)
        analysis_state["micro_step"] = micro_step

        values = []
        for key, value in analysis_state.items():
            if isinstance(value, paddle.Tensor):
                analysis_state[key] = value.item()
            values.append(analysis_state[key])

        # print(analysis_state)
        #self.state[analysis_state["sname"]].append(analysis_state)
        self.state[self.get_structure_name(pname)].append(values)


fp8_quantize_analysis_helper = FP8QuantizeAnalysisHelper()


def get_fp8_quantize_analysis_helper():
    global fp8_quantize_analysis_helper
    return fp8_quantize_analysis_helper


def can_fp8_quantize_analysis(fwd=True):
    global fp8_quantize_analysis_helper
    if fwd:
        return not is_in_recompute_phase() and fp8_quantize_analysis_helper.is_enable()
    else:
        return fp8_quantize_analysis_helper.is_enable()
