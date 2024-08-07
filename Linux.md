# Linux

## jetson orin

### io控制

```bash
# 配置io引脚功能
sudo /opt/nvidia/jetson-io/jetson-io.py

sudo apt-get install python3-pip
git clone https://github.com/NVIDIA/jetson-gpio
sudo mv jetson-gpio /opt/nvidia/
cd /opt/nvidia/jetson-gpio
sudo python3 setup.py installWS

# 添加组
sudo groupadd -f -r gpio
sudo usermod -a -G gpio jetson



sudo cp /opt/nvidia/jetson-gpio/lib/python/Jetson/GPIO/99-gpio.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger


```

## 网络

### Proxy

#### 设置apt代理

```bash

sudo vim /etc/systemd/system/docker.service.d/http-proxy.conf

# Add proxy
Acquire::http::Proxy "http://localhost:7890";
Acquire::https::Proxy "http://localhost:7890";

sudo apt update

```

#### 设置docker拉取镜像时的代理

```bash

sudo vim /etc/systemd/system/docker.service.d/http-proxy.conf

# Add proxy
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:7890"
Environment="HTTPS_PROXY=http://127.0.0.1:7890"

# Restart the service
sudo systemctl daemon-reload
sudo systemctl restart docker

docker pull 
```
