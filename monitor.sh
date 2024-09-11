#!/bin/bash

nohup python3 main.py start  > maxkb.log 2>&1 &
# 启动监听器
start_listener() {
    nohup python3 main.py start > maxkb.log 2>&1 &
    echo $! > /var/run/my_listener.pid
}
 
# 重启监听器
restart_listener() {
    kill `cat /var/run/my_listener.pid` || true
    start_listener
}
 
# 检查监听器是否运行
is_running() {
    kill -0 `cat /var/run/my_listener.pid` 2>/dev/null
}
 
# 主逻辑
start_listener
while true; do
    sleep 1 # 每10秒检查一次
    if ! is_running; then
        echo "Listener has stopped. Restarting..."
        restart_listener
    fi
done







