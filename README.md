# Multi-Agent Project Assistant

这是一个完整可运行的多 Agent 项目推进助手 Demo，适合用于展示“AI Agent / 多 Agent 协作”项目成果。

## 功能

- 需求分析 Agent：解析 PRD，提取核心需求
- 任务拆解 Agent：拆解研发、前端、测试任务
- 工期评估 Agent：结合团队规模和历史项目估算工作量
- 风险识别 Agent：识别接口依赖、安全权限、范围膨胀等风险
- 进度汇报 Agent：生成项目进展摘要和下一步动作

## 安装

```bash
pip install -r requirements.txt
```

## 运行 API 服务

```bash
uvicorn multi_agent_project_assistant:app --reload
```

打开：

```text
http://127.0.0.1:8000/docs
```

## 运行本地 Demo

```bash
python multi_agent_project_assistant.py
```

## 使用真实 LLM，可选

没有 API Key 也可以运行，会使用规则兜底逻辑。

如需接入 OpenAI-compatible API：

```bash
export OPENAI_API_KEY="your_key"
export OPENAI_MODEL="gpt-4.1-mini"
```

Windows PowerShell：

```powershell
$env:OPENAI_API_KEY="your_key"
$env:OPENAI_MODEL="gpt-4.1-mini"
```

## 示例请求

```json
{
  "project_name": "会员积分系统升级",
  "prd_text": "本次需求需要支持用户积分累计、积分兑换、积分过期提醒，并接入订单系统。用户下单后需要根据订单金额自动发放积分。积分兑换需要校验库存、用户权限和活动有效期。后台需要支持运营配置兑换规则和查看积分流水报表。",
  "team_size": 6,
  "sprint_days": 10,
  "historical_projects": [
    {"name": "优惠券系统", "estimated_days": 35, "actual_days": 42},
    {"name": "会员等级系统", "estimated_days": 28, "actual_days": 30}
  ]
}
```
