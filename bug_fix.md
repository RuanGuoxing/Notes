# bugs

## Tools

### ubuntu vscode无法输入中文

应用商店装的vscode有问题，卸载重装

```bash
sudo snap remove code

wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
rm -f packages.microsoft.gpg
sudo apt install apt-transport-https
sudo apt update
sudo apt install code

```

### 腾讯会议ubuntn不兼容wayland协议报错

```bash
sudo vi /opt/wemeet/wemeetapp.sh
```


在if [ "$XDG_SESSION_TYPE" = "wayland" ]前面加上：


```bash
export XDG_SESSION_TYPE=x11

export QT_QPA_PLATFORM=xcb

unset WAYLAND_DISPLAYCOPY
```