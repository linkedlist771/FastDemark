#!/bin/bash
# /home/ljj/.cache  models downloaded here.
# 首先应用兼容性补丁

# 然后运行原始启动脚本的内容
# 你需要把原来 start_server.sh 的内容复制到这里，
# 但在 Python 命令前加上我们的补丁

iopaint start --model=lama --device=cuda --port=8080