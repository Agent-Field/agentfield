"""
Memory Scope Hierarchy
======================

AgentField provides four memory scopes for storing agent data:

Global Scope
------------
- Shared across all agents and sessions
- Persists until explicitly deleted
- Use for: Configuration, shared knowledge bases, cross-agent state

Session Scope  
-------------
- Scoped to a single user session (conversation)
- Cleared when session ends
- Use for: Conversation context, user preferences within a session

Actor Scope
-----------
- Scoped to a single agent across all sessions
- Persists across sessions
- Use for: Agent-specific learned data, agent configuration

Workflow Scope (Run Scope)
--------------------------
- Scoped to a single workflow execution
- Cleared when workflow completes
- Use for: Intermediate results, execution-specific state

Hierarchy Diagram
-----------------
    Global (widest)
        |
    Session
        |
    Actor
        |
    Workflow/Run (narrowest)

Example Usage
-------------
>>> # Store in global scope
>>> await agent.memory.set("config", value, scope="global")
>>> 
>>> # Store in session scope  
>>> await agent.memory.set("context", value, scope="session")
"""
