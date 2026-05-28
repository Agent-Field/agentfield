"""Schemas for deep research agent with recursive planning."""

from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


class Subtask(BaseModel):
    """A single subtask in the research plan."""

    task_id: str = Field(description="Unique identifier for this task")
    description: str = Field(
        description="Clear description of what needs to be researched"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of task_ids that must complete before this task can start",
    )
    depth: int = Field(
        default=0, description="Recursion depth - how many levels deep this task is"
    )
    can_parallelize: bool = Field(
        default=True,
        description="Whether this task can run in parallel with others (if no dependencies)",
    )


class SubtaskList(BaseModel):
    """Wrapper for a list of subtasks - used for AI schema validation."""

    tasks: List[Subtask] = Field(description="List of subtasks")


class TaskDescriptions(BaseModel):
    """Simple schema for task descriptions - just a list of strings."""

    descriptions: List[str] = Field(
        description="List of task descriptions, one per line. Each should be a clear, specific research task."
    )


class TaskDependencies(BaseModel):
    """Dependency mapping for a single task."""

    task_id: str = Field(description="The task ID")
    depends_on: List[str] = Field(
        default_factory=list,
        description="List of task IDs that must complete before this task can start",
    )


class TaskDependenciesList(BaseModel):
    """Wrapper for a list of task dependencies - used for AI schema validation."""

    dependencies: List[TaskDependencies] = Field(
        description="List of task dependency mappings"
    )


class ResearchPlan(BaseModel):
    """Complete research plan with topological task graph."""

    research_question: str = Field(description="The original research question")
    tasks: List[Subtask] = Field(description="All subtasks in the plan")
    max_depth: int = Field(
        description="Maximum recursion depth reached during planning"
    )
    total_tasks: int = Field(description="Total number of tasks in the plan")
    parallelizable_groups: List[List[str]] = Field(
        default_factory=list,
        description="Groups of task_ids that can be executed in parallel",
    )


class TaskResult(BaseModel):
    """Result from executing a single research task."""

    task_id: str
    description: str
    findings: str = Field(description="Research findings for this task")
    sources: List[str] = Field(
        default_factory=list, description="Sources or references used"
    )
    confidence: str = Field(description="Confidence level: 'high', 'medium', or 'low'")


class ResearchReport(BaseModel):
    """Final synthesized research report."""

    research_question: str
    executive_summary: str = Field(description="High-level summary of findings")
    detailed_findings: List[TaskResult] = Field(
        description="Detailed findings for each task"
    )
    conclusions: str = Field(description="Final conclusions and recommendations")
    overall_confidence: str = Field(
        description="Overall confidence: 'high', 'medium', or 'low'"
    )


class SearchQueries(BaseModel):
    """Simple schema for search queries - list of strings."""

    queries: List[str] = Field(
        description="List of search query strings optimized for web search"
    )


class Citation(BaseModel):
    """Citation for a source with URL, title, and excerpt."""

    url: str = Field(description="Source URL")
    title: str = Field(description="Source title")
    excerpt: str = Field(description="Relevant excerpt from the source")


class ResearchFindings(BaseModel):
    """Structured research findings with numbered points and citations."""

    findings: str = Field(
        description="Research findings as numbered points (1. ... 2. ... etc.) with inline citations"
    )
    citations: List[Citation] = Field(
        default_factory=list, description="List of citations referenced in findings"
    )
    confidence: str = Field(description="Confidence level: 'high', 'medium', or 'low'")


class TaskMerge(BaseModel):
    """Task merge instruction for deduplication."""

    keep_task_id: str = Field(description="Task ID to keep (most specific)")
    merge_task_ids: List[str] = Field(
        description="Task IDs to merge into the kept task"
    )


class TaskMergeList(BaseModel):
    """List of task merges for deduplication."""

    merges: List[TaskMerge] = Field(
        default_factory=list, description="List of task merge instructions"
    )


class SearchStrategy(BaseModel):
    """Search strategy decision for a parent task."""

    strategy: str = Field(
        description="Strategy: 'synthesize_only' or 'enhanced_search'"
    )
    reasoning: str = Field(
        default="",
        description="Brief reasoning for the strategy choice",
    )
