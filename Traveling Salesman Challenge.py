import hashlib
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from tsp_solver import TSPInstance, TourResult, load_tsplib_file, parse_tsplib_text, solve_tsp


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "TSP Challenge"
EFFORT_LEVELS = {
	"Quick": (4, 5, 6),
	"Balanced (recommended)": (8, 15, 12),
	"Thorough": (24, 30, 20),
}
CONSTRUCTION_METHODS = ("Nearest Neighbor", "Cheapest Insertion")
IMPROVEMENT_METHODS = ("2-opt", "Swap", "Relocate", "Mixed neighborhoods", "Tabu Search")

st.set_page_config(
	page_title="TSP Route Lab",
	page_icon="",
	layout="wide",
	initial_sidebar_state="expanded",
)
st.title("TSP Route Lab")
st.caption("Choose a construction heuristic, then improve its route with local-search moves.")

available_files = sorted(DATA_DIR.glob("*.tsp"))
dataset_labels = {path.name: path for path in available_files}


def read_selected_instance(
	source: str,
	selected_path: Path | None,
	uploaded_file: object | None,
) -> tuple[TSPInstance, str]:
	if source == "Bundled instance":
		if selected_path is None:
			raise ValueError("No bundled instances are available.")
		return load_tsplib_file(selected_path), selected_path.name
	if uploaded_file is None:
		raise ValueError("Choose a .tsp file to continue.")
	content = uploaded_file.getvalue().decode("utf-8-sig")
	return parse_tsplib_text(content, uploaded_file.name), uploaded_file.name


with st.sidebar:
	st.header("Problem")
	source = st.radio("Instance source", ("Bundled instance", "Upload .tsp"), horizontal=True)
	uploaded_file = None
	selected_path = None
	if source == "Bundled instance":
		if not available_files:
			st.error("No bundled .tsp files were found in TSP Challenge/.")
		else:
			selected_name = st.selectbox("Dataset", tuple(dataset_labels))
			selected_path = dataset_labels[selected_name]
	else:
		uploaded_file = st.file_uploader("TSPLIB coordinate file", type=("tsp",))

	selected_instance = None
	selected_instance_label = None
	if selected_path is not None or uploaded_file is not None:
		try:
			selected_instance, selected_instance_label = read_selected_instance(
				source, selected_path, uploaded_file
			)
		except (UnicodeDecodeError, OSError, ValueError) as error:
			st.error(str(error))

	st.header("Build a starting tour")
	construction_method = st.selectbox("Construction heuristic", CONSTRUCTION_METHODS)
	if construction_method == "Nearest Neighbor":
		st.caption("Repeatedly visit the closest unvisited city; effort levels try different first cities.")
	else:
		st.caption("Start with a wide triangle, then insert each remaining city at the cheapest position.")

	st.header("Improve the tour")
	improvement_method = st.selectbox("Improvement method", IMPROVEMENT_METHODS, index=3)
	method_descriptions = {
		"2-opt": "Reconnect two edges by reversing the route segment between them.",
		"Swap": "Exchange the positions of two cities in the visiting order.",
		"Relocate": "Remove one city and insert it elsewhere in the visiting order.",
		"Mixed neighborhoods": "Try 2-opt, swap, and relocate moves; accept the best shortening move each step.",
		"Tabu Search": "Use all three move types and temporarily allow worse steps to escape local minima.",
	}
	st.caption(method_descriptions[improvement_method])

	st.header("Search effort")
	effort = st.radio("Choose a search level", tuple(EFFORT_LEVELS), index=1)
	starts, max_moves, tabu_tenure = EFFORT_LEVELS[effort]
	if construction_method == "Nearest Neighbor":
		st.caption(f"Tries up to {starts} starting cities, then accepts up to {max_moves} moves.")
	else:
		st.caption(f"Builds one insertion tour, then accepts up to {max_moves} moves.")
	if improvement_method == "Tabu Search":
		st.caption(f"Tabu tenure: {tabu_tenure} moves. Tabu search can take longer.")
	st.caption("More effort usually takes longer and may find a shorter route. No heuristic guarantees the optimum.")
	run_solver = st.button("Find a shorter route", type="primary", use_container_width=True)


def current_instance_key() -> str | None:
	if selected_path is not None:
		return selected_path.name
	if uploaded_file is None:
		return None
	return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


if run_solver:
	if selected_instance is None:
		st.error("Choose or upload a valid instance before running the search.")
	else:
		with st.spinner("Building and improving the route..."):
			result = solve_tsp(
				selected_instance,
				construction_method=construction_method,
				improvement_method=improvement_method,
				starts=starts,
				max_improvement_moves=max_moves,
				tabu_tenure=tabu_tenure,
			)
		st.session_state["tsp_result"] = {
			"key": (current_instance_key(), construction_method, improvement_method, starts, max_moves, tabu_tenure),
			"instance": selected_instance,
			"result": result,
		}


stored = st.session_state.get("tsp_result")
current_key = (
	current_instance_key(),
	construction_method,
	improvement_method,
	starts,
	max_moves,
	tabu_tenure,
)
if stored and stored["key"] == current_key:
	instance: TSPInstance = stored["instance"]
	result: TourResult = stored["result"]
	improvement = (
		(result.initial_length - result.length) / result.initial_length * 100
		if result.initial_length
		else 0.0
	)
	first_metric, second_metric, third_metric, fourth_metric = st.columns(4)
	first_metric.metric("Tour length", f"{result.length:,}")
	second_metric.metric("Improvement", f"{improvement:.2f}%", f"-{result.initial_length - result.length:,} units")
	third_metric.metric("Solve time", f"{result.elapsed_seconds:.2f} s")
	fourth_metric.metric("Cities", f"{len(instance.cities):,}")

	ordered_cities = [instance.cities[index] for index in result.tour]
	closed_route = ordered_cities + [ordered_cities[0]]
	figure = go.Figure()
	figure.add_trace(
		go.Scatter(
			x=[city.x for city in closed_route],
			y=[city.y for city in closed_route],
			mode="lines",
			line={"color": "#168c83", "width": 1.5},
			hoverinfo="skip",
			name="Tour",
		)
	)
	figure.add_trace(
		go.Scatter(
			x=[city.x for city in ordered_cities],
			y=[city.y for city in ordered_cities],
			mode="markers",
			marker={"color": "#d8623b", "size": 6},
			text=[str(city.node_id) for city in ordered_cities],
			hovertemplate="Node %{text}<extra></extra>",
			name="Cities",
		)
	)
	figure.update_layout(
		title=f"{instance.name} · {result.construction_method} + {result.improvement_method}",
		xaxis_title="X coordinate",
		yaxis_title="Y coordinate",
		yaxis={"scaleanchor": "x", "scaleratio": 1},
		template="plotly_white",
		height=650,
		margin={"l": 20, "r": 20, "t": 55, "b": 20},
		legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 1, "xanchor": "right"},
	)
	st.plotly_chart(
		figure,
		use_container_width=True,
		config={"displayModeBar": True, "displaylogo": False, "scrollZoom": True},
	)
	st.caption(
		f"Construction starts tried: {result.starts_tried} · "
		f"Improvement moves accepted: {result.improvement_moves} · "
		"Heuristic result; optimality is not guaranteed."
	)
else:
	st.info("Choose an instance and run the search to see its route.")
