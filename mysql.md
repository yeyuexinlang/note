### 启动 $MySQL$ 

`sudo mysql -u root -p`

- **sudo**: 管理员权限。
- **mysql**: 调用 $MySQL$ 终端。
- **-u root**: 以 $root$ 的身份登录。
- **-p**: 输入密码。

### 关闭 $MySQL$

#### 关闭 $MySQL$ 服务

**在 $MySQL$ 命令行**

`SHUTDOWN` 命令(需要为 $root$ 用户)

**通用**

`sudo systemctl stop mysql`

- **sudo**: 管理员权限。
- **systemctl stop mysql**: 让系统停止 **MySQL** 服务。

#### 退出 $MySQL$ 界面

`exit;`


