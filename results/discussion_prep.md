# 周日讨论备料（DiffSCM + CausalDiffAE）

一句话：Apr11 重跑 + 物理约束都做完了，采样侧四个约束都跑通；CausalDiffAE 的 preprocess 搭了图无关版，就等 causal graph。

## 一、已完成（汇报点）

- **数据**：Apr11 全部接入，189 文件 / 14638 场景零跳过；碰撞标签用几何 SAT 生成（2203 撞 / 12435 不撞）；修了 scene_key 跨文件重名的坑（改用 source_file:scene_key）。
- **分类器**：之前不收敛是输入没归一化（世界坐标量级几百），加了 per-feature 标准化后 PR-AUC 0.99。
- **采样侧物理约束**：preservation（原有）+ acceleration + turn angle + map，四个能同时跑、梯度都生效、碰撞推动力没被压垮。
- **训练侧**：低噪声 acceleration 正则，让模型自己采的轨迹超标率 50%→31%，且 fidelity（val 0.054 ≈ baseline 0.052）不掉；naive 全程正则会翻车（这条是个有用的方法论教训）。
- **主结果（PDF 里那版）**：正则模型 + 采样 gs5/ls2/la10，guided 超物理上限步 ~12%，Δcollision ~0.29。

> 注：发群的 PDF 只覆盖到 acceleration。turn + map 是 PDF 之后加的，**刚验证能跑、约束生效，但还没调参/量化**——讨论时说清楚这点。

## 二、待讨论 / 开放问题

- **turn / map 还没扫参**：lambda_turn、lambda_map 用的默认值，"超转角率 / 出界率"还没量化进 summarize。要不要先扫一轮再下结论。
- **map 约束是近似**：用的是原始数据 agent-centric 的 224×224 局部栅格 + 原始 raster_from_world，生成位置偏移大时靠 border padding；够不够用、要不要更精细。
- **不是严格 counterfactual**：轨迹版从随机噪声出发，没有 Abduction（对真实未来做 DDIM inversion），更准确说是 "Diff-SCM-inspired"。要不要补 trajectory inversion 做成真正的 counterfactual——这点正好和接下来上 Causal 相关。
- **训练正则还有空间**：fidelity 余量到 ~0.08，正则权重 / 低噪声窗口可以再加重，reference 超标率应该还能再降。

## 三、CausalDiffAE 下一步（Baohua 第②条）

- **preprocess**：图无关版已搭好（`preprocess_causaldiffae.py`）——每个场景导出 14 个候选因果因子（collision、collision_timestep、min_distance、closing、ego/adv 的 speed/accel/turn、path_len 等）+ 占位 DAG。图定了只要选节点 + 填邻接矩阵。
- **dataloader**：可以马上搭图无关版（按 scene_id 把轨迹 + 因子表拼起来给 CausalDiffAE），唯一依赖图的是"用哪几个因子当节点 + DAG"，做成可配置。

**需要对齐（讨论时问清）：**

- Baohua / Yongjie：那个**初步 causal graph** 的节点和边具体是什么？（是不是就在上面这些因子里选）
- Ziheng：**CausalDiffAE 代码要求的输入格式**？（他说刚跑通——通常要 样本 X + 每样本因果因子标签 + DAG 邻接矩阵）
- 分工：preprocess/dataloader 我和 Ziheng 怎么分。

## 四、日程

周日 11am 讨论。我周日有个眼部小手术，可能赶不上——材料和进度我先整理好放群里，缺席的话异步同步、回头补讨论。

## 五、可以现场展示的东西

- `diffscm_group_report.html / .pdf`（结果 + 两张图）
- `acceleration_compare.png`（无约束 vs 有约束，最直观）
- `trajectories.png`（反事实碰撞例子）
- `causaldiffae_factors.csv`（候选因果因子表，给 causal graph 讨论用）
