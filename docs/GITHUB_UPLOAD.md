# Upload this repository to GitHub

这份说明针对第一次上传 GitHub 的情况。当前目录已经是一个 Git 仓库，通常不需要再次运行 `git init`。

## 1. 创建空仓库

在 GitHub 网页上选择 `New repository`，填写仓库名，例如 `64nodes-resistor-recovery`。创建时不要勾选 README、`.gitignore` 或 License，因为本地已经有这些文件。

## 2. 在本地检查

打开 PowerShell：

```powershell
cd E:\64Nodes
git status --short
git remote -v
```

本仓库的 `private_materials/`、大型 CSV、cache、outputs、模型权重和本地 vendor 已配置为忽略。可以额外确认：

```powershell
git status --short --ignored
git check-ignore private_materials\终稿\README.md
git check-ignore data\training_data64Nodes_2.csv
git check-ignore gnn\GNN_CLS\modelo3\outputs\model_last.pt
```

三个 `git check-ignore` 命令都应该返回对应的忽略规则。

## 3. 暂存并检查文件清单

如果你确认要上传当前项目的全部公开代码和文档：

```powershell
git add -A
git status --short
git diff --cached --stat
git diff --cached --name-only | Select-String "private_materials|终稿|答辩|诚信|外文"
```

最后一条命令应该没有输出。如果暂存内容不对，可以只取消暂存，不会删除文件：

```powershell
git restore --staged .
```

如果只想先提交本次项目管理文档，可以改用：

```powershell
git add .gitignore README.md CONTRIBUTING.md LICENSE requirements.txt docs data\README.md scripts\README.md Figure\README.md history\README.md gnn mlp inverse_identifiability square_scale_study\README.md
```

## 4. 提交本地版本

```powershell
git commit -m "Organize repository for public release"
```

如果 Git 提示没有设置作者信息，只需设置一次：

```powershell
git config --global user.name "你的 GitHub 显示名"
git config --global user.email "你的 GitHub 邮箱"
```

然后重新运行 `git commit`。

## 5. 绑定并推送到 GitHub

把下面 URL 替换成你在 GitHub 创建的仓库地址：

```powershell
git branch -M main
git remote add origin https://github.com/你的用户名/你的仓库名.git
git push -u origin main
```

第一次通过 HTTPS 推送时，Git 可能打开浏览器让你登录。GitHub 不再接受普通账户密码作为 Git 密码；按 Git Credential Manager 的浏览器登录流程完成即可。

## 6. 网页确认

推送完成后刷新 GitHub 仓库页面，检查：

- 根目录 README 能正常打开；
- `private_materials/` 不在文件列表中；
- 没有 `.pt`、`.pth`、大型 CSV 或个人信息文档；
- `gnn/`、`mlp/`、`docs/` 的 README 链接可以打开。

以后修改项目时，只需重复：

```powershell
git add -A
git commit -m "说明本次修改"
git push
```

不要使用 `git add -f private_materials/...`，也不要把论文终稿、答辩 PPT 或个人信息文件拖到 GitHub 网页上传。
