import os
import re
import csv
import ast
from collections import defaultdict, deque

SOURCE_WDIR = os.path.join("D:\\", "dev", "experiments")
OUT_WDIR = os.path.join("D:\\", "dev", "EstimatingInspectionTasks")

def extract_block(text: str, start_label: str) -> str:
    pattern = rf"{re.escape(start_label)}\s*(.*?)(?=\n[A-Za-z_][A-Za-z0-9_]*:\s|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_value(text: str, key: str) -> str:
    pattern = rf"^{re.escape(key)}:\s*(.*)$"
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def flatten_digraph(digraph_text: str) -> str:
    return " ".join(line.strip() for line in digraph_text.splitlines())


def parse_fit_value(fit_values_text: str) -> float:
    try:
        values = ast.literal_eval(fit_values_text)
        if isinstance(values, list) and values:
            return float(values[0])
    except Exception:
        pass
    return float("nan")


def parse_nodes_and_edges(digraph_text: str):
    node_pattern = re.compile(r'([A-Za-z0-9_]+)\s*\[label="(.*?)"\];', re.DOTALL)
    edge_pattern = re.compile(r'([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)\s*\[\];')

    nodes = {}
    edges = []

    for node_id, label in node_pattern.findall(digraph_text):
        nodes[node_id] = label

    for src, dst in edge_pattern.findall(digraph_text):
        edges.append((src, dst))

    return nodes, edges


def operator_name_from_label(label: str) -> str:
    return label.split("\\n")[0].split("\n")[0].strip()


def parameter_count_from_label(label: str) -> int:
    lines = label.split("\\n") if "\\n" in label else label.split("\n")
    if not lines:
        return 0
    # first line = operator name, remaining lines = parameters
    return sum(1 for line in lines[1:] if "=" in line)


def compute_pipeline_depth(nodes, edges) -> int:
    graph = defaultdict(list)
    indegree = defaultdict(int)

    for node_id in nodes:
        indegree[node_id] = 0

    for src, dst in edges:
        graph[src].append(dst)
        indegree[dst] += 1

    # longest path in DAG using topological traversal
    queue = deque()
    dist = {}

    for node_id in nodes:
        if indegree[node_id] == 0:
            queue.append(node_id)
            dist[node_id] = 1

    visited = 0
    max_depth = 0

    while queue:
        current = queue.popleft()
        visited += 1
        max_depth = max(max_depth, dist[current])

        for neighbor in graph[current]:
            dist[neighbor] = max(dist.get(neighbor, 1), dist[current] + 1)
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    # exclude HalconInputNode from depth count if present
    operator_depth = max_depth
    if any(operator_name_from_label(label) == "HalconInputNode" for label in nodes.values()):
        operator_depth -= 1

    return max(operator_depth, 0)


def derive_short_name(dataset: str) -> str:
    parts = dataset.split("_")
    return parts[-1] if parts else dataset


def derive_long_name(dataset: str) -> str:
    return f"{dataset}_mean_pipeline"


def analyze_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    dataset = extract_value(text, "dataset")
    fit_values_text = extract_value(text, "fit_values")
    digraph_text = extract_block(text, "digraph:")

    mcc = parse_fit_value(fit_values_text)
    digraph_flat = flatten_digraph(digraph_text)

    nodes, edges = parse_nodes_and_edges(digraph_text)

    operator_nodes = []
    total_parameters = 0

    for node_id, label in nodes.items():
        op_name = operator_name_from_label(label)
        if op_name == "HalconInputNode":
            continue
        operator_nodes.append(op_name)
        total_parameters += parameter_count_from_label(label)

    number_of_operators = len(operator_nodes)
    pipeline_depth = compute_pipeline_depth(nodes, edges)
    number_of_unique_operators = len(set(operator_nodes))

    return {
        "long": derive_long_name(dataset),
        "short": derive_short_name(dataset),
        "MCC": mcc,
        "number_of_operators": number_of_operators,
        "pipeline_depth": pipeline_depth,
        "number_of_unique_operators": number_of_unique_operators,
        "total_number_of_parameters": total_parameters,
        "digraph": digraph_flat,
    }


def merge_pipeline_data(search_path: str, output_csv: str):
    txt_files = [
        os.path.join(search_path, fn)
        for fn in os.listdir(search_path)
        if fn.lower().endswith(".txt")
    ]

    rows = []
    for filepath in txt_files:
        try:
            rows.append(analyze_file(filepath))
        except Exception as e:
            print(f"Skipping {filepath} due to error: {e}")

    fieldnames = [
        "long",
        "short",
        "MCC",
        "number_of_operators",
        "pipeline_depth",
        "number_of_unique_operators",
        "total_number_of_parameters",
        "digraph",
    ]

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(
            f,
            delimiter=";",
            quoting=csv.QUOTE_ALL  # or QUOTE_MINIMAL (less safe)
        )
        writer.writerow(fieldnames)

        for row in rows:
            writer.writerow([
                row["long"],
                row["short"],
                f"{row['MCC']:.17f}" if row["MCC"] == row["MCC"] else "",
                row["number_of_operators"],
                row["pipeline_depth"],
                row["number_of_unique_operators"],
                row["total_number_of_parameters"],
                row["digraph"],
            ])

    print(f"CSV written to: {output_csv}")

def main():
    digraph_mean = os.path.join(SOURCE_WDIR, "param_tuning", "digraph_mean")
    digraph_best = os.path.join(SOURCE_WDIR, "param_tuning", "digraph_best")

    pipeline_mean_summary_path = os.path.join(OUT_WDIR, "", "pipeline_mean_summary.csv")
    pipeline_best_summary_path = os.path.join(OUT_WDIR, "", "pipeline_best_summary.csv")

    print("Merging pipeline data from digraph files...")
    merge_pipeline_data(digraph_mean, pipeline_mean_summary_path)

    print("Merging pipeline data from MCC files...")
    merge_pipeline_data(digraph_best, pipeline_best_summary_path)

if __name__ == "__main__":
    main()