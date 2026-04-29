"""
Multi-Agent Project Assistant

A complete, runnable demo that simulates a multi-agent project management assistant.
It covers:
1. Requirement analysis from PRD text
2. Task breakdown
3. Engineering effort estimation
4. Risk identification
5. Progress summary generation
6. Optional LLM integration via OpenAI-compatible API

How to run:
    pip install -r requirements.txt
    uvicorn multi_agent_project_assistant:app --reload

Then open:
    http://127.0.0.1:8000/docs

Environment variables, optional:
    OPENAI_API_KEY=your_key
    OPENAI_MODEL=gpt-4.1-mini

If no API key is provided, the app will use deterministic rule-based fallback logic.
"""

from __future__ import annotations

import os
import re
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ProjectRequest(BaseModel):
    project_name: str = Field(..., examples=["会员积分系统升级"])
    prd_text: str = Field(..., examples=["本次需求需要支持用户积分累计、积分兑换、过期提醒，并接入订单系统。"])
    team_size: int = Field(default=5, ge=1, le=100)
    sprint_days: int = Field(default=10, ge=1, le=90)
    historical_projects: Optional[List[Dict[str, Any]]] = Field(default=None)


class RequirementItem(BaseModel):
    title: str
    description: str
    priority: Priority


class TaskItem(BaseModel):
    id: str
    title: str
    owner_role: str
    description: str
    estimated_days: float
    dependencies: List[str] = Field(default_factory=list)


class RiskItem(BaseModel):
    title: str
    level: RiskLevel
    reason: str
    mitigation: str


class ProjectPlan(BaseModel):
    project_id: str
    project_name: str
    created_at: str
    requirements: List[RequirementItem]
    tasks: List[TaskItem]
    risks: List[RiskItem]
    total_estimated_days: float
    recommended_sprint_count: int
    progress_summary: str
    next_actions: List[str]


class LLMClient:
    """OpenAI-compatible client with rule-based fallback."""

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.client = OpenAI(api_key=self.api_key) if self.api_key and OpenAI else None

    def available(self) -> bool:
        return self.client is not None

    def complete_json(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        if not self.available():
            return None

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"
        return json.loads(content)


class BaseAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        parts = re.split(r"[。！？\n；;.!?]", text)
        return [p.strip() for p in parts if p.strip()]


class RequirementAnalysisAgent(BaseAgent):
    def run(self, prd_text: str) -> List[RequirementItem]:
        system_prompt = """
你是一个资深产品经理。请从 PRD 中提取核心需求，输出 JSON。
JSON 格式：
{
  "requirements": [
    {"title": "", "description": "", "priority": "high|medium|low"}
  ]
}
只输出 JSON。
"""
        user_prompt = f"PRD 内容：\n{prd_text}"
        result = self.llm.complete_json(system_prompt, user_prompt)

        if result and "requirements" in result:
            return [RequirementItem(**item) for item in result["requirements"]]

        return self._fallback(prd_text)

    def _fallback(self, prd_text: str) -> List[RequirementItem]:
        sentences = self.split_sentences(prd_text)
        keywords_high = ["支付", "订单", "权限", "安全", "风控", "核心", "必须", "上线"]
        keywords_medium = ["提醒", "通知", "统计", "配置", "报表", "搜索"]

        requirements: List[RequirementItem] = []
        for idx, sentence in enumerate(sentences[:8], start=1):
            priority = Priority.LOW
            if any(k in sentence for k in keywords_high):
                priority = Priority.HIGH
            elif any(k in sentence for k in keywords_medium):
                priority = Priority.MEDIUM

            requirements.append(
                RequirementItem(
                    title=f"需求 {idx}: {sentence[:18]}",
                    description=sentence,
                    priority=priority,
                )
            )

        if not requirements:
            requirements.append(
                RequirementItem(
                    title="需求澄清",
                    description="PRD 信息不足，需要补充业务目标、用户范围、核心流程和验收标准。",
                    priority=Priority.HIGH,
                )
            )
        return requirements


class TaskBreakdownAgent(BaseAgent):
    def run(self, requirements: List[RequirementItem]) -> List[TaskItem]:
        system_prompt = """
你是一个技术项目经理。请基于需求拆解研发任务，输出 JSON。
JSON 格式：
{
  "tasks": [
    {
      "title": "",
      "owner_role": "Frontend|Backend|QA|PM|Designer|Data",
      "description": "",
      "estimated_days": 1.5,
      "dependencies": []
    }
  ]
}
只输出 JSON。
"""
        user_prompt = json.dumps(
            {"requirements": [r.model_dump() for r in requirements]},
            ensure_ascii=False,
        )
        result = self.llm.complete_json(system_prompt, user_prompt)

        if result and "tasks" in result:
            tasks = []
            for item in result["tasks"]:
                tasks.append(
                    TaskItem(
                        id=str(uuid.uuid4())[:8],
                        title=item["title"],
                        owner_role=item.get("owner_role", "Backend"),
                        description=item.get("description", ""),
                        estimated_days=float(item.get("estimated_days", 1.0)),
                        dependencies=item.get("dependencies", []),
                    )
                )
            return tasks

        return self._fallback(requirements)

    def _fallback(self, requirements: List[RequirementItem]) -> List[TaskItem]:
        tasks: List[TaskItem] = []
        for req in requirements:
            base_days = 2.0 if req.priority == Priority.HIGH else 1.5 if req.priority == Priority.MEDIUM else 1.0
            backend_id = str(uuid.uuid4())[:8]
            frontend_id = str(uuid.uuid4())[:8]

            tasks.append(
                TaskItem(
                    id=backend_id,
                    title=f"后端实现：{req.title}",
                    owner_role="Backend",
                    description=f"设计接口、数据模型和业务逻辑，覆盖：{req.description}",
                    estimated_days=base_days,
                    dependencies=[],
                )
            )
            tasks.append(
                TaskItem(
                    id=frontend_id,
                    title=f"前端交互：{req.title}",
                    owner_role="Frontend",
                    description=f"完成页面交互、状态处理和异常提示，覆盖：{req.description}",
                    estimated_days=max(1.0, base_days - 0.5),
                    dependencies=[backend_id],
                )
            )
            tasks.append(
                TaskItem(
                    id=str(uuid.uuid4())[:8],
                    title=f"测试验收：{req.title}",
                    owner_role="QA",
                    description=f"补充正常流程、异常流程和边界条件测试，覆盖：{req.description}",
                    estimated_days=1.0,
                    dependencies=[backend_id, frontend_id],
                )
            )
        return tasks


class EngineeringEstimationAgent(BaseAgent):
    def run(
        self,
        tasks: List[TaskItem],
        team_size: int,
        sprint_days: int,
        historical_projects: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        complexity_factor = self._estimate_complexity_factor(historical_projects)
        raw_days = sum(task.estimated_days for task in tasks)
        adjusted_days = round(raw_days * complexity_factor, 1)

        team_capacity_per_sprint = max(team_size * sprint_days * 0.65, 1)
        sprint_count = int((adjusted_days + team_capacity_per_sprint - 1) // team_capacity_per_sprint)
        sprint_count = max(sprint_count, 1)

        return {
            "total_estimated_days": adjusted_days,
            "recommended_sprint_count": sprint_count,
            "complexity_factor": complexity_factor,
        }

    def _estimate_complexity_factor(self, historical_projects: Optional[List[Dict[str, Any]]]) -> float:
        if not historical_projects:
            return 1.15

        overruns = []
        for project in historical_projects:
            estimated = float(project.get("estimated_days", 1))
            actual = float(project.get("actual_days", estimated))
            if estimated > 0:
                overruns.append(actual / estimated)

        if not overruns:
            return 1.15

        avg = sum(overruns) / len(overruns)
        return round(min(max(avg, 0.9), 1.5), 2)


class RiskIdentificationAgent(BaseAgent):
    def run(self, prd_text: str, requirements: List[RequirementItem], tasks: List[TaskItem]) -> List[RiskItem]:
        system_prompt = """
你是一个项目风险管理专家。请识别项目风险并输出 JSON。
JSON 格式：
{
  "risks": [
    {"title": "", "level": "high|medium|low", "reason": "", "mitigation": ""}
  ]
}
只输出 JSON。
"""
        user_prompt = json.dumps(
            {
                "prd_text": prd_text,
                "requirements": [r.model_dump() for r in requirements],
                "tasks": [t.model_dump() for t in tasks],
            },
            ensure_ascii=False,
        )
        result = self.llm.complete_json(system_prompt, user_prompt)

        if result and "risks" in result:
            return [RiskItem(**item) for item in result["risks"]]

        return self._fallback(prd_text, requirements, tasks)

    def _fallback(self, prd_text: str, requirements: List[RequirementItem], tasks: List[TaskItem]) -> List[RiskItem]:
        risks: List[RiskItem] = []
        text = prd_text.lower()

        if any(k in prd_text for k in ["第三方", "外部", "接入", "支付", "订单", "接口"]):
            risks.append(
                RiskItem(
                    title="外部系统依赖风险",
                    level=RiskLevel.HIGH,
                    reason="需求涉及外部系统或接口接入，接口变更、联调排期和数据口径可能影响交付。",
                    mitigation="提前锁定接口协议、准备 mock 服务，并将联调时间纳入排期。",
                )
            )

        if any(k in prd_text for k in ["权限", "安全", "风控", "隐私", "合规"]):
            risks.append(
                RiskItem(
                    title="安全与权限风险",
                    level=RiskLevel.HIGH,
                    reason="需求涉及权限、安全或敏感数据，若设计不完整可能带来线上风险。",
                    mitigation="在设计评审阶段加入安全检查清单，并增加越权、异常输入和审计日志测试。",
                )
            )

        if len(requirements) >= 5 or len(tasks) >= 15:
            risks.append(
                RiskItem(
                    title="范围膨胀风险",
                    level=RiskLevel.MEDIUM,
                    reason="需求点和任务数量较多，容易出现范围扩大或优先级不清。",
                    mitigation="按 P0/P1/P2 拆分上线范围，优先保证主链路交付。",
                )
            )

        if "待定" in prd_text or "不明确" in prd_text or "todo" in text:
            risks.append(
                RiskItem(
                    title="需求不确定风险",
                    level=RiskLevel.MEDIUM,
                    reason="PRD 中存在待定或不明确内容，可能导致返工。",
                    mitigation="在开发前补充验收标准，并建立需求变更记录。",
                )
            )

        if not risks:
            risks.append(
                RiskItem(
                    title="测试覆盖不足风险",
                    level=RiskLevel.LOW,
                    reason="若只覆盖主流程，边界条件和异常场景可能遗漏。",
                    mitigation="为每个核心需求补充边界 case、异常 case 和回归测试清单。",
                )
            )

        return risks


class ReportingAgent(BaseAgent):
    def run(
        self,
        project_name: str,
        requirements: List[RequirementItem],
        tasks: List[TaskItem],
        risks: List[RiskItem],
        estimation: Dict[str, Any],
    ) -> Dict[str, Any]:
        high_risks = [risk for risk in risks if risk.level == RiskLevel.HIGH]
        backend_count = len([task for task in tasks if task.owner_role.lower() == "backend"])
        frontend_count = len([task for task in tasks if task.owner_role.lower() == "frontend"])
        qa_count = len([task for task in tasks if task.owner_role.lower() == "qa"])

        summary = (
            f"项目《{project_name}》已完成自动化拆解：共识别 {len(requirements)} 个核心需求，"
            f"拆解出 {len(tasks)} 个执行任务，其中后端 {backend_count} 个、前端 {frontend_count} 个、测试 {qa_count} 个。"
            f"预计总工作量约 {estimation['total_estimated_days']} 人日，建议拆分为 "
            f"{estimation['recommended_sprint_count']} 个 Sprint 推进。"
            f"当前识别到 {len(risks)} 个风险点，其中高风险 {len(high_risks)} 个。"
        )

        next_actions = [
            "组织一次需求澄清会，确认 P0 范围和验收标准。",
            "提前确认外部接口、数据口径和联调负责人。",
            "为高风险模块补充技术方案评审和专项测试用例。",
            "每日自动生成进展摘要，跟踪延期任务和新增风险。",
        ]

        if high_risks:
            next_actions.insert(0, "优先处理高风险项，明确负责人和截止时间。")

        return {
            "progress_summary": summary,
            "next_actions": next_actions,
        }


class ProjectAssistantOrchestrator:
    def __init__(self) -> None:
        llm = LLMClient()
        self.requirement_agent = RequirementAnalysisAgent(llm)
        self.task_agent = TaskBreakdownAgent(llm)
        self.estimation_agent = EngineeringEstimationAgent(llm)
        self.risk_agent = RiskIdentificationAgent(llm)
        self.reporting_agent = ReportingAgent(llm)

    def run(self, request: ProjectRequest) -> ProjectPlan:
        requirements = self.requirement_agent.run(request.prd_text)
        tasks = self.task_agent.run(requirements)
        estimation = self.estimation_agent.run(
            tasks=tasks,
            team_size=request.team_size,
            sprint_days=request.sprint_days,
            historical_projects=request.historical_projects,
        )
        risks = self.risk_agent.run(request.prd_text, requirements, tasks)
        report = self.reporting_agent.run(
            project_name=request.project_name,
            requirements=requirements,
            tasks=tasks,
            risks=risks,
            estimation=estimation,
        )

        return ProjectPlan(
            project_id=str(uuid.uuid4()),
            project_name=request.project_name,
            created_at=datetime.now().isoformat(timespec="seconds"),
            requirements=requirements,
            tasks=tasks,
            risks=risks,
            total_estimated_days=estimation["total_estimated_days"],
            recommended_sprint_count=estimation["recommended_sprint_count"],
            progress_summary=report["progress_summary"],
            next_actions=report["next_actions"],
        )


app = FastAPI(
    title="Multi-Agent Project Assistant",
    description="A multi-agent assistant for requirement analysis, task breakdown, risk tracking and progress reporting.",
    version="1.0.0",
)

orchestrator = ProjectAssistantOrchestrator()


@app.get("/")
def health_check() -> Dict[str, str]:
    return {
        "status": "ok",
        "message": "Multi-Agent Project Assistant is running. Visit /docs to try the API.",
    }


@app.post("/project/plan", response_model=ProjectPlan)
def create_project_plan(request: ProjectRequest) -> ProjectPlan:
    if not request.prd_text.strip():
        raise HTTPException(status_code=400, detail="prd_text cannot be empty")
    return orchestrator.run(request)


if __name__ == "__main__":
    demo_request = ProjectRequest(
        project_name="会员积分系统升级",
        prd_text=(
            "本次需求需要支持用户积分累计、积分兑换、积分过期提醒，并接入订单系统。"
            "用户下单后需要根据订单金额自动发放积分。"
            "积分兑换需要校验库存、用户权限和活动有效期。"
            "后台需要支持运营配置兑换规则和查看积分流水报表。"
        ),
        team_size=6,
        sprint_days=10,
        historical_projects=[
            {"name": "优惠券系统", "estimated_days": 35, "actual_days": 42},
            {"name": "会员等级系统", "estimated_days": 28, "actual_days": 30},
        ],
    )

    plan = orchestrator.run(demo_request)
    print(plan.model_dump_json(indent=2))
