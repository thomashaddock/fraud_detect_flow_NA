"""Anomaly detection crew for financial document processing.

Two-agent crew: anomaly analyst + risk assessor. No tools -- data is
passed via YAML interpolation from the flow state. The crew focuses
purely on analysis and risk assessment.
"""

from crewai import Agent, Crew, Process, Task, LLM
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------

class AnomalyDetail(BaseModel):
    """A single anomaly finding."""

    record_index: str = Field(description="Row index or record ID affected.")
    description: str = Field(description="What the anomaly is.")
    severity: str = Field(description="low, medium, high, or critical.")


class AnomalyAnalysis(BaseModel):
    """Structured output from the analyst's anomaly scan."""

    anomalies_detected: bool = Field(description="Whether any anomalies were found.")
    anomaly_count: int = Field(default=0, description="Number of anomalies found.")
    anomaly_details: list[AnomalyDetail] = Field(default_factory=list, description="List of anomaly objects.")
    preliminary_risk_score: int = Field(default=0, description="Risk score 0-100.")
    summary: str = Field(default="", description="2-3 sentence summary of findings.")


class AnomalyVerdict(BaseModel):
    """Final structured output from the risk assessor."""

    anomalies_detected: bool = Field(description="Whether anomalies were confirmed.")
    risk_score: int = Field(default=0, description="Final risk score 0-100.")
    anomaly_details: list[AnomalyDetail] = Field(default_factory=list, description="Confirmed anomaly list.")
    recommendation: str = Field(description="PASS, MANUAL_REVIEW, or FAIL.")
    explanation: str = Field(default="", description="2-3 sentence explanation.")


# ---------------------------------------------------------------------------
# Crew definition
# ---------------------------------------------------------------------------

@CrewBase
class AnomalyCrew:
    """Two-agent anomaly detection crew (analyst + risk assessor).

    Receives file data and validation results through YAML interpolation.
    No tools needed -- pure analysis.
    """

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def anomaly_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["anomaly_analyst"],  # type: ignore[index]
            tools=[],
            verbose=True,
            max_iter=10,
            llm=LLM(model="openai/gpt-4o-mini", temperature=0.1),
        )

    @agent
    def risk_assessor(self) -> Agent:
        return Agent(
            config=self.agents_config["risk_assessor"],  # type: ignore[index]
            tools=[],
            verbose=True,
            max_iter=10,
            llm=LLM(model="openai/gpt-4o-mini", temperature=0.0),
        )

    @task
    def analyze_anomalies(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_anomalies"],  # type: ignore[index]
            output_pydantic=AnomalyAnalysis,
        )

    @task
    def assess_risk(self) -> Task:
        return Task(
            config=self.tasks_config["assess_risk"],  # type: ignore[index]
            output_pydantic=AnomalyVerdict,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
