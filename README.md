# 垃圾邮件审核 · 模型版本固定 + 截止时间感知批处理 + 全局接单上限

FastAPI + NumPy 的垃圾邮件审核运维控制台。网页可切换现行模型、并发提交至多 16 封邮件
（每封独立 100–1000ms 截止时间），单工作器按 **模型版本** 组批推理，逐条展示分数、分类、
模型版本与批次。全局最多同时接单 **8 封**：同步与异步提交共用名额，超出部分逐封标记为
「过载拒单」且不进入模型队列。

## 核心契约

| 需求 | 实现位置 |
| --- | --- |
| 全局最多接单 8 封（同步/异步共用） | `Worker.max_admitted`（`MAX_ADMITTED_EMAILS`，默认 8）；`Worker.admit()` 在接单瞬间同步判定 |
| 逐封显示接单 / 过载 | 每个 `JobEntry` 的 `admission` + `status="rejected"`；网页「接单」列与任务头「拒单 N」 |
| 拒单不进入模型队列 | 拒单发生在 `WorkItem` 入队之前，不建队列项、不占批次、无 `batch_id` |
| 排队邮件到期释放名额 | 队列内定时器触发 `_finalize` → `_release_lease` |
| 超时后打分仍在跑则继续占名额 | 打分包在独立 task 中，名额只在真实计算完成回调里释放；`timed_out_still_executing` 可观测 |
| 已接单邮件仍按原模型与各自截止结算 | 版本在接单时快照；批内并发打分、逐封独立判定 |
| 排队邮件按「接单时」的版本分类 | 网关 `_intake_entry` 接单时快照 `active_version` 写入请求；工作器只读该字段 |
| 网页切换现行模型 | `POST /api/model` + 前端模型卡片 |
| 并发提交 ≤16 封、每封 100–1000ms 截止 | `SubmitIn.emails`（`max_length=16` / `ge=100, le=1000`）|
| 逐条查看分数/分类/版本/批次 | 结果表 + `GET /api/jobs/{id}` |
| 单工作器按版本组批，每批 ≤4、等待 ≤200ms | `app/batching.py` 的 `_plan` / `Worker` 默认参数 |
| 新请求立即用新版，未发批的旧请求用旧版 | 版本在接单时固定；批次按版本分区 |
| 批内不混版本 | 队列按版本分桶，`_plan` 每组批只取同一版本 |
| 最早一封快到截止时间时提前发批 | `dispatch_guard_ms`（默认 25ms）提前派发 |
| 过期请求不占后续批次 | 每个请求有独立定时器，到期即解析并从队列移除 |
| 迟到结果不得改判成功 | 批次内 `wait_for` 按各自剩余时间截断 + 完成后显式比对截止时间 |
| 同批其他邮件按各自截止时间返回 | 批内并发打分，逐条独立判定/独立完成 |

## 快速开始

### Docker Compose（推荐）

```bash
docker compose up --build
# 打开 http://localhost:8000
```

镜像内已固定 `models/v1.json` 与 `models/v2.json` 两份不同的英文词袋线性模型，同时以只读卷
挂载 `./models` 便于查看。Compose 只起 **一个** 副本、uvicorn 也只有 1 个 worker——批次调度、
接单名额与「现行模型」指针都是进程内状态。

### 本地运行

```bash
python3 -m pip install -r requirements.txt
python3 scripts/build_models.py          # 重新生成模型（可选，仓库已含产物）
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### 测试

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest -q                     # 28 项：调度语义 + 接单上限 + 网关行为
python3 scripts/selfcheck.py             # 对运行中的服务做端到端自测（默认 :8000）
```

## 模型

两份模型都是 `bow-linear.v1` 格式（冻结词表 + IDF + 线性权重 + 偏置 + 阈值），推理只用
NumPy，网关与测试结果逐位一致。

| 版本 | 说明 | 词表 | 阈值 |
| --- | --- | --- | --- |
| `v1` | 2024 baseline，只有通用促销词 | 27 | 0.50 |
| `v2` | 2026 hardened，加入 crypto/parcel/urgency 词，权重重拟合 | 63 | 0.60 |

同一封「加密货币投资」邮件在 v1 下是 ham（0.20 左右），在 v2 下是 spam（0.87 左右），
所以版本固定能在页面上直接看出来。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 审核控制台 |
| GET | `/healthz` | 健康检查 + 现行模型 |
| GET | `/api/models` | 模型列表（含权重摘要） |
| POST | `/api/model` | 切换现行模型 `{"version": "v2"}` |
| GET | `/api/state` | 工作器指标、接单名额快照、按版本统计、最近批次、最近任务 |
| POST | `/api/jobs` | 异步提交（202），返回任务与逐封接单进度 |
| GET | `/api/jobs/{id}` | 任务结果（含逐封 `admission`） |
| POST | `/api/classify` | 同步提交并等待结果，与异步共用接单名额 |
| GET | `/api/batches` | 最近批次记录 |
| POST | `/api/demo/interleaved` | 一键交错演示：v1 接单 → 切 v2 → 新请求 + 慢推理 |

提交体示例：

```json
{
  "label": "人工抽检",
  "emails": [
    {"text": "Congratulations! You are a winner.", "deadline_ms": 900},
    {"text": "Hi team, attached is the quarterly report.", "deadline_ms": 300,
     "submit_after_ms": 50, "simulate_latency_ms": 0}
  ]
}
```

`submit_after_ms`（0–5000）与 `simulate_latency_ms`（0–5000）是演示用：前者错峰接单，
后者模拟慢推理，用来观察「批次执行中超时 + 迟到结果丢弃」。二者只影响时点，不改变批处理
契约。注意 `submit_after_ms` 决定的是 **申请名额的时刻**：错峰提交如果在那一刻名额已满，
该封邮件就会被拒单。

## 调度语义细节

- 队列按版本分桶；每轮只派发 **最早到期** 的那一批，避免一个慢批次吞掉另一个版本的截止时间。
- 组批规则：同版本队列满 4 封立即发批；不足 4 封时等 `min(最老入队 + 200ms, 最早截止 − 25ms)`。
  已经等满 200ms 的请求优先成批（防饿死），剩余名额用同版本 EDF 工作填充。
- 派发时同步从队列摘除并取消该请求的定时器，执行阶段由 `asyncio.wait_for(剩余时间)` 与
  完成后的显式截止检查共同判定；超时结果先到就先解析，迟到结果只做
  `late_results_discarded` 标记，**不会** 把 `timeout` 改回 `ok`。
- 批内并发打分，因此单封慢邮件不会拖住同批其他邮件；每封各自按自己的截止时间返回。

## 接单名额（全局 8 封）

- 名额是 **每封邮件** 一个，池子全局唯一：`POST /api/jobs` 与 `POST /api/classify` 共用。
- 接单判定在入队之前同步完成，因此并发提交不会超卖。拿不到名额的邮件得到
  `status="rejected"` / `reason="admission_overloaded"`，**不会** 创建队列项、不会进入任何批次，
  也不计入 `submitted`；它单独计入 `rejected_total`。
- 名额何时释放：
  - 仍在排队时到期：定时器解析为 `timeout` 并立即释放；
  - 已派发执行：名额转为 `executing` 并一直持有到 **真实打分调用返回**。即使调用方已经拿到
    `timeout`（截止时间先到、迟到结果被丢弃），不可中断的打分线程仍占着名额，计算结束后才可接新单。
- `GET /api/state` 的 `admission` 字段：

  | 字段 | 含义 |
  | --- | --- |
  | `limit` / `used` / `available` | 上限、当前占用、剩余可接 |
  | `queued` / `executing` | 排队中 / 执行中的名额占用 |
  | `timed_out_still_executing` | 已报超时但打分仍在运行、仍占名额的数量 |
  | `admitted_total` / `rejected_total` | 进程内累计接单 / 累计拒单 |

## 已知限制

- 状态全部在进程内（单工作器要求），重启会丢失进行中的任务与切换历史；没有持久化。
- 模型产物是仓库内预置的两份静态 JSON，没有训练流水线；切换只改指针，不做热加载校验。
- 演示用的 `simulate_latency_ms` 通过 `asyncio.to_thread` 中的 `sleep` 模拟，结束的线程
  不会被强制中断（Python 线程无法抢占），只是其结果被丢弃。因此「超时但仍占名额」的时长等于
  该线程真实跑完的时长；这是有意的限流行为，不是泄漏。
- 接单上限默认 8，可用 `MAX_ADMITTED_EMAILS` 调整；单次提交上限仍是 16 封（`MAX_EMAILS_PER_SUBMISSION`），
  当上限小于提交量时，同一提交的邮件会逐封被拒。
