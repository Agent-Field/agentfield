"""Recursive planning router for deep research agent."""

from __future__ import annotations

import asyncio
from typing import List

from agentfield import AgentRouter

from schemas import (
    ResearchPlan,
    ResearchReport,
    SearchStrategy,
    Subtask,
    TaskDescriptions,
    TaskDependenciesList,
    TaskMergeList,
    TaskResult,
)


planning_router = AgentRouter(prefix="planning")


def _build_topological_groups(tasks: List[Subtask]) -> List[List[str]]:
    """
    Build topological groups using topological sort.

    Groups tasks by dependency level - all tasks at the same level
    can run in parallel. Returns list of groups, where each group
    contains task IDs that can execute in parallel.
    """
    task_map = {task.task_id: task for task in tasks}

    # Calculate in-degree for each task (how many dependencies it has)
    in_degree = {task_id: len(task.dependencies) for task_id, task in task_map.items()}

    # Topological sort - group by level
    level_groups: List[List[str]] = []
    remaining = set(task_map.keys())

    while remaining:
        # Find all tasks with no remaining dependencies (current level)
        current_level = [task_id for task_id in remaining if in_degree[task_id] == 0]

        if not current_level:
            # Circular dependency or error - add remaining tasks
            current_level = list(remaining)

        level_groups.append(current_level)
        remaining -= set(current_level)

        # Decrease in-degree for tasks that depend on current level
        for task_id in current_level:
            for dep_task in tasks:
                if task_id in dep_task.dependencies and dep_task.task_id in remaining:
                    in_degree[dep_task.task_id] -= 1

    # Return groups with more than one task (parallelizable)
    return [group for group in level_groups if len(group) > 1]


def _get_execution_levels(tasks: List[Subtask]) -> List[List[str]]:
    """
    Get all execution levels using topological sort.

    Returns ALL levels (not just parallelizable groups).
    Level 0 = leaf tasks (no dependencies, need search)
    Level 1+ = parent tasks (have dependencies, synthesize from children)
    """
    task_map = {task.task_id: task for task in tasks}

    # Calculate in-degree for each task
    in_degree = {task_id: len(task.dependencies) for task_id, task in task_map.items()}

    # Topological sort - group by level
    level_groups: List[List[str]] = []
    remaining = set(task_map.keys())

    while remaining:
        # Find all tasks with no remaining dependencies (current level)
        current_level = [task_id for task_id in remaining if in_degree[task_id] == 0]

        if not current_level:
            # Circular dependency or error - add remaining tasks
            current_level = list(remaining)

        level_groups.append(current_level)
        remaining -= set(current_level)

        # Decrease in-degree for tasks that depend on current level
        for task_id in current_level:
            for dep_task in tasks:
                if task_id in dep_task.dependencies and dep_task.task_id in remaining:
                    in_degree[dep_task.task_id] -= 1

    return level_groups


@planning_router.reasoner()
async def create_research_plan(
    research_question: str,
    max_depth: int = 3,
    max_tasks_per_level: int = 5,
) -> ResearchPlan:
    """
    Create a recursive research plan by breaking down a question into subtasks.

    This reasoner recursively decomposes a research question into smaller,
    manageable subtasks, forming a topological graph that can be executed
    in parallel where dependencies allow.
    """
    # First, get initial breakdown from LLM - simplified to just descriptions
    initial_response = await planning_router.ai(
        system=(
            "You are an expert research planner breaking down questions into a hierarchical task graph.\n\n"
            "## PHILOSOPHY\n"
            "You are creating two types of tasks:\n"
            "1. **Leaf tasks** (no dependencies) = Specific questions that need web search\n"
            "   - These are atomic research questions answerable via web search\n"
            "   - Example: 'What is AgentField?' or 'What are AgentField's features?'\n\n"
            "2. **Parent tasks** (have dependencies) = Questions answered by synthesizing children\n"
            "   - These combine answers from child tasks\n"
            "   - Example: 'Summarize AgentField' depends on 'What is AgentField?' and 'What are features?'\n\n"
            "## CRITICAL: DISTINCT TASKS\n"
            "Each task must be DISTINCT and NON-OVERLAPPING.\n"
            "- Do NOT create tasks that ask the same question in different words\n"
            "- Each task should cover a unique aspect or angle\n"
            "- Avoid redundancy - if two tasks would find the same information, merge them\n\n"
            "## DEPENDENCY MEANING\n"
            "A task depends on another if it needs that task's ANSWER to proceed.\n"
            "Dependencies mean: 'I need answers from those tasks to answer this.'\n\n"
            "## YOUR TASK\n"
            "Break the research question into 3-5 DISTINCT major research areas.\n"
            "Each area should be a specific, searchable question (leaf task).\n"
            "Ensure each task is unique and non-overlapping.\n"
            "These will be refined recursively, and synthesis tasks will be created later.\n\n"
            f"Maximum {max_tasks_per_level} tasks.\n\n"
            "Return ONLY a JSON object with a 'descriptions' array."
        ),
        user=(
            f"Research Question: {research_question}\n\n"
            f"Break this into 3-5 DISTINCT major research areas as specific, searchable questions. "
            f"Ensure each task is unique and covers different aspects. Return only the descriptions array."
        ),
        schema=TaskDescriptions,
    )

    # Build initial Subtask objects from descriptions
    initial_plan = [
        Subtask(
            task_id=f"task_{idx + 1}",
            description=desc,
            dependencies=[],
            depth=1,
            can_parallelize=True,
        )
        for idx, desc in enumerate(initial_response.descriptions)
    ]

    # Recursively refine each task
    all_tasks: List[Subtask] = []

    async def refine_recursively(
        task: Subtask, parent_context: str = ""
    ) -> List[Subtask]:
        """Recursively refine a task by calling the refine_task reasoner."""
        if task.depth >= max_depth:
            # Base case: task is specific enough
            return [task]

        # Directly call the refine_task reasoner - this creates a workflow node
        refined_tasks = await refine_task(
            task_description=task.description,
            parent_context=parent_context or research_question,
            current_depth=task.depth,
            max_depth=max_depth,
        )

        # Update task IDs to maintain hierarchy and recursively refine
        result_tasks: List[Subtask] = []
        for idx, refined_task in enumerate(refined_tasks):
            new_task_id = f"{task.task_id}_{idx + 1}"
            new_task = Subtask(
                task_id=new_task_id,
                description=refined_task.description,
                dependencies=refined_task.dependencies,
                depth=task.depth + 1,
                can_parallelize=len(refined_task.dependencies) == 0,
            )

            # Recursively refine if still not at max depth
            if new_task.depth < max_depth:
                further_refined = await refine_recursively(new_task, task.description)
                result_tasks.extend(further_refined)
            else:
                result_tasks.append(new_task)

        # If no refinement happened, return the original task
        return result_tasks if result_tasks else [task]

    # Refine initial tasks - can run in parallel since they're independent
    refinement_tasks = [
        refine_recursively(initial_task, research_question)
        for initial_task in initial_plan
    ]
    refined_results = await asyncio.gather(*refinement_tasks)

    # Flatten the results
    for refined_list in refined_results:
        all_tasks.extend(refined_list)

    # Deduplicate similar/redundant tasks
    tasks_for_dedup = [
        {"task_id": task.task_id, "description": task.description} for task in all_tasks
    ]
    merge_response = await deduplicate_tasks(tasks_for_dedup, research_question)

    # Apply merges: keep the keep_task_id, remove merge_task_ids, update dependencies
    task_map = {task.task_id: task for task in all_tasks}
    tasks_to_remove = set()

    for merge in merge_response.merges:
        if merge.keep_task_id in task_map:
            # Update dependencies: if any task depends on a merged task, point to kept task
            for task in all_tasks:
                if merge.keep_task_id != task.task_id:
                    # Replace merged task IDs with kept task ID in dependencies
                    if merge.keep_task_id in task.dependencies:
                        # Already depends on kept task, remove merged ones
                        task.dependencies = [
                            dep
                            for dep in task.dependencies
                            if dep not in merge.merge_task_ids
                        ]
                    else:
                        # Replace merged task IDs with kept task ID
                        task.dependencies = [
                            merge.keep_task_id if dep in merge.merge_task_ids else dep
                            for dep in task.dependencies
                        ]

            # Mark merged tasks for removal
            tasks_to_remove.update(merge.merge_task_ids)

    # Remove merged tasks
    all_tasks = [task for task in all_tasks if task.task_id not in tasks_to_remove]

    # Identify dependencies between tasks
    tasks_for_analysis = [
        {"task_id": task.task_id, "description": task.description} for task in all_tasks
    ]
    dependencies_response = await identify_dependencies(
        tasks_for_analysis, research_question
    )

    # Update task dependencies
    task_map = {task.task_id: task for task in all_tasks}
    dependency_map = {
        dep.task_id: dep.depends_on for dep in dependencies_response.dependencies
    }

    for task in all_tasks:
        if task.task_id in dependency_map:
            task.dependencies = dependency_map[task.task_id]
            # Validate dependencies exist
            task.dependencies = [dep for dep in task.dependencies if dep in task_map]
            task.can_parallelize = len(task.dependencies) == 0

    # Build topological groups using topological sort
    parallelizable_groups = _build_topological_groups(all_tasks)

    return ResearchPlan(
        research_question=research_question,
        tasks=all_tasks,
        max_depth=max(task.depth for task in all_tasks) if all_tasks else 0,
        total_tasks=len(all_tasks),
        parallelizable_groups=parallelizable_groups,
    )


@planning_router.reasoner()
async def refine_task(
    task_description: str,
    parent_context: str = "",
    current_depth: int = 0,
    max_depth: int = 3,
) -> List[Subtask]:
    """
    Recursively refine a single task into smaller subtasks.

    This is a helper reasoner that can be called recursively to break down
    complex tasks into more manageable pieces. Each call creates a workflow node.
    """

    if current_depth >= max_depth:
        # Return the task as-is if we've hit max depth
        return [
            Subtask(
                task_id=f"leaf_{current_depth}",
                description=task_description,
                dependencies=[],
                depth=current_depth,
                can_parallelize=True,
            )
        ]

    # Use AI to determine if task needs further breakdown - simplified schema
    result_response = await planning_router.ai(
        system=(
            "You are a task decomposition expert. Break down research tasks into "
            "smaller subtasks.\n\n"
            "## CRITICAL: DISTINCT SUBTASKS\n"
            "When breaking down, create DISTINCT, NON-OVERLAPPING subtasks:\n"
            "- Each subtask should cover a unique aspect\n"
            "- Do NOT create subtasks that ask the same question\n"
            "- Ensure each subtask is independently researchable\n\n"
            "Return ONLY a JSON object with a 'descriptions' array.\n"
            "- If the task is specific enough, return it as a single-item array\n"
            "- If it needs breakdown, return 2-4 DISTINCT smaller task descriptions\n"
            "- Each description must be unique and non-overlapping\n\n"
            f"Current depth: {current_depth} of {max_depth}. "
            f"Only break down if still too complex."
        ),
        user=(
            f"Task: {task_description}\n"
            f"{f'Context: {parent_context}' if parent_context else ''}\n\n"
            f"Break this into DISTINCT smaller tasks if needed, or return as-is if specific enough. "
            f"Ensure each subtask is unique and non-overlapping."
        ),
        schema=TaskDescriptions,
    )

    # Build Subtask objects from descriptions
    refined_tasks = [
        Subtask(
            task_id=f"subtask_{idx}",
            description=desc,
            dependencies=[],
            depth=current_depth + 1,
            can_parallelize=True,
        )
        for idx, desc in enumerate(result_response.descriptions)
    ]

    return refined_tasks


@planning_router.reasoner()
async def identify_dependencies(
    tasks: List[dict], research_question: str
) -> TaskDependenciesList:
    """
    Identify dependencies between tasks based on their descriptions.

    Takes a list of tasks with id and description, and identifies which
    tasks need results from other tasks before they can execute.
    """
    # Format tasks for the prompt
    tasks_text = "\n".join(
        f"- {task['task_id']}: {task['description']}" for task in tasks
    )

    response = await planning_router.ai(
        system=(
            "You are a dependency analysis expert identifying task relationships.\n\n"
            "## DEPENDENCY PHILOSOPHY\n"
            "A task depends on another if it needs that task's ANSWER to proceed.\n"
            "Dependencies mean: 'I need answers from those tasks to answer this.'\n\n"
            "## CRITICAL: AVOID REDUNDANT DEPENDENCIES\n"
            "- Do NOT mark dependencies if tasks ask the same question\n"
            "- Do NOT mark dependencies if a task can be answered independently\n"
            "- Only mark dependencies when a task truly needs another task's answer\n"
            "- If two tasks would find the same information, they should NOT depend on each other\n\n"
            "## DEPENDENCY PATTERNS\n"
            "- **Comparison tasks** depend on individual research tasks\n"
            "  Example: 'Compare X and Y' depends on 'What is X?' and 'What is Y?'\n\n"
            "- **Synthesis tasks** depend on all component tasks\n"
            "  Example: 'Summarize AgentField' depends on 'What is AgentField?' and 'What are features?'\n\n"
            "- **Analysis tasks** depend on data-gathering tasks\n"
            "  Example: 'Analyze trends' depends on 'What are the trends?'\n\n"
            "## KEY PRINCIPLE\n"
            "If a task can be answered by combining other tasks' findings, mark dependencies.\n"
            "Leaf tasks (no dependencies) will be answered via web search.\n"
            "Parent tasks (have dependencies) will synthesize from children's answers.\n\n"
            "Return ONLY a JSON object with a 'dependencies' array. Each item:\n"
            "- task_id: the task ID\n"
            "- depends_on: array of task IDs it depends on (empty if independent/leaf task)\n\n"
            "Be conservative - only mark dependencies when clearly needed. "
            "Avoid redundant dependencies between similar tasks."
        ),
        user=(
            f"Research Question: {research_question}\n\n"
            f"Tasks:\n{tasks_text}\n\n"
            f"Identify dependencies. For each task, determine if it needs answers from other tasks. "
            f"DO NOT mark dependencies if tasks ask the same question or can be answered independently. "
            f"Return dependency mappings for all tasks."
        ),
        schema=TaskDependenciesList,
    )

    return response


@planning_router.reasoner()
async def deduplicate_tasks(tasks: List[dict], research_question: str) -> TaskMergeList:
    """
    Identify and merge similar/redundant tasks.

    Finds tasks that ask the same question or cover the same information,
    and suggests which tasks to merge into others.
    """
    tasks_text = "\n".join(
        f"- {task['task_id']}: {task['description']}" for task in tasks
    )

    response = await planning_router.ai(
        system=(
            "You are a task deduplication expert. Identify tasks that are redundant or ask the same question.\n\n"
            "## YOUR TASK\n"
            "Find tasks that:\n"
            "- Ask the same question in different words\n"
            "- Cover the same information/research area\n"
            "- Are essentially duplicates\n\n"
            "For each group of similar tasks:\n"
            "- Keep the most specific/clear task (keep_task_id)\n"
            "- Mark others to merge (merge_task_ids)\n\n"
            "## IMPORTANT\n"
            "- Only merge tasks that are truly redundant\n"
            "- Keep tasks that cover different aspects, even if related\n"
            "- Be conservative - when in doubt, don't merge\n\n"
            "Return ONLY a JSON object with a 'merges' array. "
            "Each merge has keep_task_id and merge_task_ids."
        ),
        user=(
            f"Research Question: {research_question}\n\n"
            f"Tasks:\n{tasks_text}\n\n"
            f"Identify redundant tasks and suggest merges. Return merge instructions."
        ),
        schema=TaskMergeList,
    )

    return response


@planning_router.reasoner()
async def decide_search_strategy(
    task_description: str,
    research_question: str,
    dependency_findings: List[TaskResult],
) -> SearchStrategy:
    """
    Decide whether a parent task should:
    - synthesize_only: Answer from children's findings alone
    - enhanced_search: Need additional web search with dependency context

    Examples:
    - "Summarize X" when we have "What is X?" → synthesize_only
    - "Compare X to Y" when we have "What is X?" → enhanced_search (need Y info)
    """
    children_text = "\n".join(
        f"- {dep.task_id}: {dep.description}\n  Answer: {dep.findings[:200]}..."
        for dep in dependency_findings
    )

    response = await planning_router.ai(
        system=(
            "You are a task strategy expert deciding how to answer a parent task.\n\n"
            "## TWO STRATEGIES\n"
            "1. **synthesize_only**: Can answer from children's findings alone\n"
            "   - Example: 'Summarize X' when we have 'What is X?' and 'What are X's features?'\n"
            "   - Example: 'List all features' when we have individual feature tasks\n\n"
            "2. **enhanced_search**: Needs additional web search with dependency context\n"
            "   - Example: 'Compare X to Y' when we have 'What is X?' but need Y info\n"
            "   - Example: 'Analyze market trends' when we have data but need external context\n"
            "   - Example: 'Find competitors' when we have product info but need market research\n\n"
            "## DECISION CRITERIA\n"
            "- If the task asks for information NOT in children → enhanced_search\n"
            "- If the task can be answered by combining children → synthesize_only\n"
            "- If the task needs external context/comparison → enhanced_search\n"
            "- If the task is pure synthesis/summary → synthesize_only\n\n"
            "Return ONLY a JSON object with 'strategy' ('synthesize_only' or 'enhanced_search') "
            "and 'reasoning' (brief explanation)."
        ),
        user=(
            f"Research Question: {research_question}\n\n"
            f"Parent Task: {task_description}\n\n"
            f"Child Task Answers:\n{children_text}\n\n"
            f"Decide: Can we answer the parent task from children alone, or do we need additional search? "
            f"Return strategy and reasoning."
        ),
        schema=SearchStrategy,
    )

    return response


async def _execute_task_smart(
    task: Subtask,
    research_question: str,
    completed_results: dict[str, TaskResult],
) -> TaskResult:
    """
    Smart task executor that routes to search or synthesis based on dependencies and strategy.

    - Leaf tasks (no deps) → search via execute_research_task
    - Parent tasks (has deps) → decide strategy:
      - synthesize_only → synthesize_from_dependencies
      - enhanced_search → execute_research_task_with_context
    """
    from routers.research import (
        execute_research_task,
        execute_research_task_with_context,
        synthesize_from_dependencies,
    )

    if not task.dependencies:
        # Leaf task - search for answers
        return await execute_research_task(
            task_id=task.task_id,
            task_description=task.description,
            research_question=research_question,
        )
    else:
        # Parent task - get dependency findings
        dependency_findings = [
            completed_results[dep_id]
            for dep_id in task.dependencies
            if dep_id in completed_results
        ]

        # Decide strategy: synthesize_only or enhanced_search
        strategy = await decide_search_strategy(
            task_description=task.description,
            research_question=research_question,
            dependency_findings=dependency_findings,
        )

        if strategy.strategy == "synthesize_only":
            # Pure synthesis from children
            return await synthesize_from_dependencies(
                task_id=task.task_id,
                task_description=task.description,
                research_question=research_question,
                dependency_findings=dependency_findings,
            )
        else:
            # Enhanced search with dependency context
            return await execute_research_task_with_context(
                task_id=task.task_id,
                task_description=task.description,
                research_question=research_question,
                dependency_findings=dependency_findings,
            )


@planning_router.reasoner()
async def execute_deep_research(
    research_question: str,
    max_depth: int = 3,
    max_tasks_per_level: int = 5,
) -> ResearchReport:
    """
    Complete deep research orchestrator.

    Takes a research question and:
    1. Creates recursive research plan with dependencies
    2. Executes tasks in topological order (leaf tasks search, parent tasks synthesize)
    3. Synthesizes final comprehensive report
    """
    # Step 1: Create research plan
    plan = await create_research_plan(
        research_question=research_question,
        max_depth=max_depth,
        max_tasks_per_level=max_tasks_per_level,
    )

    # Step 2: Get execution levels (topological order)
    execution_levels = _get_execution_levels(plan.tasks)
    task_map = {task.task_id: task for task in plan.tasks}
    completed_results: dict[str, TaskResult] = {}

    # Step 3: Execute tasks level by level
    for level_idx, level_tasks in enumerate(execution_levels):
        # Execute tasks in parallel within each level
        execution_coroutines = [
            _execute_task_smart(
                task=task_map[task_id],
                research_question=research_question,
                completed_results=completed_results,
            )
            for task_id in level_tasks
            if task_id in task_map
        ]

        if execution_coroutines:
            level_results = await asyncio.gather(*execution_coroutines)
            for result in level_results:
                completed_results[result.task_id] = result

    # Step 4: Synthesize final report
    all_findings = list(completed_results.values())

    # Build context for final synthesis
    findings_text = "\n\n".join(
        f"Task {r.task_id}: {r.description}\n"
        f"Findings: {r.findings}\n"
        f"Sources: {', '.join(r.sources) if r.sources else 'Synthesized from child tasks'}\n"
        f"Confidence: {r.confidence}"
        for r in all_findings
    )

    synthesis = await planning_router.ai(
        system=(
            "You are a research synthesis expert creating a comprehensive final report.\n\n"
            "## CONTEXT\n"
            "You are synthesizing findings from multiple research tasks into a final report.\n"
            "Each task has been answered (either via web search for leaf tasks, or "
            "synthesis for parent tasks).\n\n"
            "## YOUR TASK\n"
            "Create a comprehensive research report that:\n"
            "1. Provides an executive summary answering the main research question\n"
            "2. Organizes detailed findings by topic/theme\n"
            "3. Draws conclusions and key insights\n"
            "4. Assesses overall confidence based on all task findings\n\n"
            "Structure the report clearly and comprehensively."
        ),
        user=(
            f"Research Question: {research_question}\n\n"
            f"All Task Findings:\n{findings_text}\n\n"
            f"Create a comprehensive research report that synthesizes all findings "
            f"to answer the research question. Include executive summary, detailed findings, "
            f"conclusions, and overall confidence assessment."
        ),
        schema=ResearchReport,
    )

    # Attach detailed findings
    synthesis.detailed_findings = all_findings

    return synthesis
