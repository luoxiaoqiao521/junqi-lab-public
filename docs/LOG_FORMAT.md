# JSONL schema 1
一行header，初始frame，逐手frame，末行必须终局且换行。完整日志属裁判私有。
event为[seat,from,to,outcome,eliminated_mask,revealed_mask]；初始为null。
seat 为南东北西 0..3。四人局按 `seat%2` 归队。单挑暗棋标记为 `match=single-dark`，南、北分别是队 0、队 1，东西不参赛。ply 为一次行动，node 为 0..128 点位。
rank包含特殊棋子，不等同普通军衔；Unknown=-1；具体编号以core/types.h为准。
truth 为裁判身份，views 为对局当时按模式过滤的四个座位视角。暗棋的研究全明只切换显示。出局原因从事件的淘汰位、碰撞结果与赛前配置还原。启用夺旗出局时，仅同一手失去真正军旗的座位标为军旗被夺。该事件的其他淘汰位为轮到时无合法着。
legal_move_counts表示各座位在该盘面的候选数量，并非四家同时行动。
result 依次为进行中、队 0 胜、队 1 胜、和棋、模拟截断。
seq等于ply。哈希为回归校验，不是安全认证；state_hash不单独证明冗余数组一致。
版本/棋盘/配置不匹配拒绝重放；新格式须显式版本迁移，不悄改旧记录。
