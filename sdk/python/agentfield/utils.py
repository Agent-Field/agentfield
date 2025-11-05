import socket


def get_free_port(start_port=8001, end_port=8999):
    """
    Find an available port in the specified range.

    Args:
        start_port (int): Start of port range
        end_port (int): End of port range

    Returns:
        int: Available port number

    Raises:
        RuntimeError: If no free port found in range
    """
    for port in range(start_port, end_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            continue

    raise RuntimeError(f"No free port found in range {start_port}-{end_port}")
