from typing import List, Dict, Any

class GraphUtils:
    @staticmethod
    def add_node(graph: Dict[str, List[str]], node: str) -> None:
        """Add a node to the graph if it doesn't already exist."""
        if node not in graph:
            graph[node] = []

    @staticmethod
    def add_edge(graph: Dict[str, List[str]], from_node: str, to_node: str) -> None:
        """Add a directed edge from from_node to to_node."""
        if from_node in graph and to_node in graph:
            graph[from_node].append(to_node)

    @staticmethod
    def remove_node(graph: Dict[str, List[str]], node: str) -> None:
        """Remove a node and all edges associated with it."""
        if node in graph:
            del graph[node]
            for edges in graph.values():
                if node in edges:
                    edges.remove(node)

    @staticmethod
    def remove_edge(graph: Dict[str, List[str]], from_node: str, to_node: str) -> None:
        """Remove a directed edge from from_node to to_node."""
        if from_node in graph and to_node in graph[from_node]:
            graph[from_node].remove(to_node)

    @staticmethod
    def get_neighbors(graph: Dict[str, List[str]], node: str) -> List[str]:
        """Return a list of neighbors for a given node."""
        return graph.get(node, [])

    @staticmethod
    def display_graph(graph: Dict[str, List[str]]) -> None:
        """Print the graph in a readable format."""
        for node, edges in graph.items():
            print(f"{node} -> {', '.join(edges) if edges else 'No edges'}")