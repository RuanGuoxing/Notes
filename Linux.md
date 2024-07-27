# Linux

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
