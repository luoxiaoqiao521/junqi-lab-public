# 四国军棋研究台

本仓库保存离线训练器、规则引擎、实验工具和交接说明。主线支持四暗、双明、全明和南北单挑，另有一个独立的 Ataraxos 单挑适配实验。本次快照对应本机研究版本 `fab51a1`。

## 查找入口

| 要做的事 | 入口 |
| --- | --- |
| 接手研究、核对当前验证到哪一步 | [项目指南](docs/PROJECT_GUIDE.md) |
| 查看裁判、观察、策略与训练接口 | [模块与证据边界](docs/PROJECT_GUIDE.md#模块索引) |
| 查看唯一下一项研究 | [续跑入口](ops/CONTINUE.md) |
| 查看单挑适配器 | [Ataraxos 对照](experiments/ataraxos/README.md) |
| 查看暂缓的子力账本与评估方案 | [研究待办](docs/PIECE_ECONOMY_BACKLOG.md) |
| 查看公开版和本机数据的边界 | [公开边界](docs/PUBLIC_SOURCE_SECURITY.md) |
| 核查修改范围、信息隔离与评估口径 | [AGENTS.md](AGENTS.md) |

## 本地构建与验证

目前使用 Windows、Visual Studio 2022 Build Tools 的 C++ 编译器、PowerShell 7、Python 3.12 和 Node.js。在仓库根目录运行：

```powershell
pwsh -NoProfile -File .\tools\check.ps1
pwsh -NoProfile -File .\start-mvp.ps1 -Seed 42 -NoOpen
```

第一条命令构建引擎，执行核心规则、观察隔离、命令行、网页逻辑及工程检查，并在 `dist/junqi-mvp` 生成本地运行包。第二条命令在本机生成一局夺旗练习，显示输出路径。默认练习使用 `flag-route` 启发式。研究模型的训练与加载通过显式研究命令进行。

此快照包含引擎和本地回放相关的通用源码。棋谱、完整对局记录、逐盘报告、模型权重、外部依赖副本及原工作树的历史留在本机。测试所需的合成对局由当前源码生成。依赖旧本机制品的训练实验需要另备对应输入指纹及调度合同，范围见项目指南。
