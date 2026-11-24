# pairwise-gapless-priming-delay

## 领域

游戏、图形与多媒体

## 主题

无缝播放预滚与编码延迟

## 说明

按实际延迟裁剪、预滚入队与设备时钟。

初始状态下 `media/gapless_priming_delay.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
