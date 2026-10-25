# -*- coding: utf-8 -*-
class TodoNotImplemented(Exception):
    """未实现标记。

    看到它不要慌 —— 这不是报错，是「这里留给你写」的路标。
    跑实验时如果撞上它，说明你把 PTQLAB_REFERENCE 关了但 TODO 还没填完。
    """

    def __init__(self, name):
        self.name = name
        super().__init__("TODO 未实现: %s" % name)
