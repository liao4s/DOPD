import re


def parse_1click_metrics(text: str=None, file_name: str=None):
    if text is None and file_name is not None:
        with open(file_name, "r", encoding="utf-8") as f:
            text = f.read()
    metrics = {}
    in_block = False
    # 用来匹配多值指标行，如 "e2e-latency(avg, P50, P90, P99): 5.64, 5.61, 6.22, 8.54"
    multi_val_pattern = re.compile(r'^(.*\))\s*:\s*(.*)$')
    all_metrics = []
    # 按行遍历字符串
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('[BeginMetrics]'):
            in_block = True
            continue
        if line.startswith('[EndMetrics]'):
            all_metrics.append(metrics)
            metrics = {}
            in_block = False
            continue
        if not in_block or not line or ':' not in line:
            continue

        # 先按第一个冒号拆 key/value
        key, raw = line.split(':', 1)
        key = key.strip()
        raw = raw.strip()

        # 检查是否匹配多值指标
        m = multi_val_pattern.match(line)
        if m:
            # m.group(1) 包括 "(avg, ...)" 部分
            k, vals = m.groups()
            k = k.strip()
            metrics[k] = [float(x) for x in vals.split(',')]
        else:
            # 单值：尝试转 float，失败则保留字符串
            try:
                metrics[key] = [float(raw) for raw in raw.split(',')]
            except ValueError:
                metrics[key] = [raw]
    sort_key = "batch-size"
    sort_index = 0
    # 方法二：就地排序
    all_metrics.sort(
        key=lambda m: m.get(sort_key, [float("inf")])[sort_index]
    )
    return all_metrics