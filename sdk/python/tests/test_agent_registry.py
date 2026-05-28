from agentfield.agent_registry import (
    set_current_agent,
    get_current_agent_instance,
    clear_current_agent,
)
import queue
import threading

class DummyAgent:
    pass


def test_agent_registry_roundtrip():
    clear_current_agent()
    assert get_current_agent_instance() is None

    agent = DummyAgent()
    set_current_agent(agent)
    assert get_current_agent_instance() is agent

    clear_current_agent()
    assert get_current_agent_instance() is None


def test_agent_registry_thread_isolation():
    clear_current_agent()
    main_thread_agent = DummyAgent()
    set_current_agent(main_thread_agent)

    result_queue: "queue.Queue[object]" = queue.Queue()
    started = threading.Event()

    def worker():
        started.set()
        result_queue.put(get_current_agent_instance())

    thread = threading.Thread(target=worker)
    thread.start()
    started.wait(timeout=2)
    thread.join(timeout=2)

    assert thread.is_alive() is False
    assert result_queue.get(timeout=2) is None
    assert get_current_agent_instance() is main_thread_agent

    clear_current_agent()


def test_agent_registry_concurrent_set():
    clear_current_agent()
    main_thread_agent = DummyAgent()
    set_current_agent(main_thread_agent)
    assert get_current_agent_instance() is main_thread_agent

    result_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
    lock = threading.Lock()
    ready_count = 0
    start_event = threading.Event()

    def worker(name: str):
        nonlocal ready_count
        local_agent = DummyAgent()
        set_current_agent(local_agent)
        with lock:
            ready_count += 1
            if ready_count == 2:
                start_event.set()
        start_event.wait(timeout=2)
        result_queue.put((name, get_current_agent_instance()))

    thread_a = threading.Thread(target=worker, args=("a",))
    thread_b = threading.Thread(target=worker, args=("b",))
    thread_a.start()
    thread_b.start()
    thread_a.join(timeout=2)
    thread_b.join(timeout=2)

    assert thread_a.is_alive() is False
    assert thread_b.is_alive() is False

    thread_results = {
        thread_name: agent_instance
        for thread_name, agent_instance in (
            result_queue.get(timeout=2),
            result_queue.get(timeout=2),
        )
    }
    assert thread_results["a"] is not thread_results["b"]
    assert thread_results["a"] is not main_thread_agent
    assert thread_results["b"] is not main_thread_agent

    clear_current_agent()
