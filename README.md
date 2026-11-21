# 防护贴老化挑批资料交换

`data/` 收录供演练使用的合成记录，不含真实企业或人员资料。资料使用 UTF-8 逗号分隔，首行为字段名，带逗号的单值用双引号括起。日期采用 ISO 表示，空字段表示来源未填或不适用，不代表数值零。`sources.json` 记录各表列名、身份键和可直接核对的引用。

## 业务资料

- `inventory.csv`：当前批次有共同生产批、数量、入库龄、合同用途限制。时间均以天为单位。
- `observations.csv`：检测窗口是贴片年龄；失效落在前次已确证正常与首次确证失效之间，未观察至终点留空。sample_prob 是该来源抽到该片的概率。
- `contracts.csv`：duration 是到达后保障时长，risk_limit 是整个使用段可接受的失效风险；family_cap 是同生产家族数量上限。
- `warranty.csv`：合同保修界线。它影响申诉覆盖，不改物理失效。
- `picks.csv`：已确认和出库的持有关系。不存在的确认不得被当作既有约束。
- `rechecks.csv`：复查窗口容量，用 lot_options 约束可处理的库存。
- `lab_options.csv`：复查动作的抽走件数、能覆盖的检查龄及结果返回时点。未来值不得当成现有事实。
- `supplemental.csv`：有实际现场回执的复查记录。received 是批备货时钟上的到达时点。
- `historical_orders.csv`：冻结的历史挑批依据和订单使用结果，shipment_age 包含在途年龄；尚未收齐的结果不当成持续完好。

last_seen 是最后能确认观察状态的龄期，seen_on 是记录可用的日期；同一试件最后收到的行替代旧观察记录。数量是可用单件数；备用抽查会实际占住这些件。保障时长从抵达日计，库存 age 以相应 available_at 为基准。合同允许混装时，可并用兼容批但仍要受同家族上限约束。

## 本地核对

需要 Python 3.11。分析环境可使用 `python3 -m pip install -r requirements.txt` 安装锁定的数值依赖。

从仓库根目录执行 `python3 check_sources.py`，返回各表行数；其他同格式资料可用 `python3 check_sources.py 路径` 检查。脚本只读资料并报告结构或引用错误，不给出业务结论。执行 `python3 -m unittest -v` 检查资料约定及拒绝重复身份的行为。
