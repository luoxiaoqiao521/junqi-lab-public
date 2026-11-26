# 碰撞判定优先级与补充的实验取值（供 P0-02）

用途：消除"炸弹与任何子同归于尽""非工兵撞雷必败"等用例之间的冲突。
以下取值是实验取值，不是 QQ 已确认规则。每个配置键只列可执行的取值，不使用"其他"。

记号：A = 攻击方棋子，D = 防守方棋子。

## 判定顺序（命中即止）

0. 前置条件（不属于碰撞）
   - D 在行营：该走法非法。
   - A 是地雷、军旗，或 A 在大本营内：没有合法走法。

1. A 或 D 是炸弹
   - 另一方是军旗：按 `combat.bomb_vs_flag`
     - `both_removed`：炸弹与军旗都移除
     - `bomb_captures`：炸弹存活并夺旗
   - 另一方是地雷：按 `combat.bomb_vs_mine`
     - `both_removed`：都移除
     - `bomb_removed_mine_stays`：炸弹移除，地雷保留
   - 其余：双方都移除。

2. D 是地雷（A 不是炸弹）
   - A 是工兵：A 存活，地雷移除。
   - A 不是工兵：A 移除；地雷按 `combat.mine_survives_non_engineer`
     - `true`：地雷保留
     - `false`：地雷也移除

3. D 是军旗（A 不是炸弹）
   - A 存活，军旗移除，进入淘汰流程。

4. 其余（双方都是行军等级：司令至工兵）
   - 等级高者存活，低者移除；等级相同则双方都移除。

## 与碰撞分开定义的后果

| 配置键 | 取值 | 实验取值 | 说明 |
|---|---|---|---|
| elim.flag_removed_eliminates | true / false | true | 军旗被移除是否导致该方出局；与"炸弹是否存活"无关 |
| elim.piece_handling | remain_as_obstacles / removed | removed | 出局方剩余棋子的处理 |
| turn.first_seat | seeded_random / fixed_seat_0 | seeded_random | 首手座位 |
| terminal.priority | win_before_draw / draw_before_win | win_before_draw | 同一手同时触发胜负与判和时的优先级 |
| marshal_death.reveal_flag | true / false | false | 司令被吃后军旗是否亮出 |

## 测试要求

- 判定顺序的每个分支、每个配置取值，至少一个用例。
- 36 组不同行军等级对，攻守双向各测；9 组同级各测一次。
- 炸弹作为攻击方与作为防守方分别测试。
- 用例名称写明所依赖的配置取值，避免两份用例在不同配置下互相矛盾。
