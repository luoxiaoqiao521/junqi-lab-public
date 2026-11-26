# Ataraxos 单挑战台

这是军棋训练器与 Ataraxos 的本地实验适配器。它用 Ataraxos 的原生 Stratego 规则和公开教师，考察 CPU 裁判接入、候选覆盖和学生的动作表示。单挑结果以该实验的规则为限，军棋课程沿用本项目的原生规则。

需要 Windows MSVC、Python 依赖（见 [本机实验版本](requirements-installed.txt)），以及在本目录 `upstream` 下预备的上游提交 `92db29e8ffc323b1b8a2804b5c3f84695d036b05`。该目录由使用者准备。裁判构建会验证其实际使用的源码文件与该提交相同。上游代码和预训练模型来自 [Ataraxos 公开仓库](https://github.com/AtaraxosAI/stratego/tree/92db29e8ffc323b1b8a2804b5c3f84695d036b05)。

在本目录运行：

```powershell
python -B build_referee.py
python -B verify_arena.py
python -B arena.py runs/local-pilot
```

最后一条需要本机上游教师权重，且输出目录须是新名称。实验输出、对局记录、现成模型及第三方源码均由本机管理。
