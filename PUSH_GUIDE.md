# 一键推送项目二到 GitHub（在你自己的机器上运行，本环境 HTTPS 出网被沙箱封锁）
# 用法: 把下面的 <你的仓库名> 换成真实仓库名，然后在本机 PowerShell 执行

# 1) 如果你还没有创建仓库，先在 GitHub 网页新建空仓库（不要勾选 README），名字建议:
#    a-share-pairs-trading

# 2) 在本地仓库目录执行（已含 2 次提交，主分支 main，工作树干净）:
cd F:\Deepseekwork\秋招\quant-pairs-strategy
git remote add origin https://github.com/Mxk-zhongqiu/<你的仓库名>.git
git push -u origin main

# 3) 如果推送时要求认证（首次）:
#    - 网页登录后使用 Personal Access Token（Settings -> Developer settings -> PAT，
#      scope 勾选 repo），密码框粘贴 token 即可
#    或使用 GitHub CLI: gh auth login && gh repo create <仓库名> --source . --push

# 说明: 仓库已 gitignore 数据/本地文件（真实行情数据不入库，红线规则）
