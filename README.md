# TSP Route Lab

A Streamlit application for visualizing approximate solutions to TSPLIB coordinate instances. Choose either Nearest Neighbor or Cheapest Insertion to construct a starting tour, then improve it using 2-opt, Swap, Relocate, Mixed neighborhoods, or Tabu Search. All construction and improvement logic is implemented in `tsp_solver.py`; no optimization library is used.

The parser supports `TYPE: TSP` instances with `EDGE_WEIGHT_TYPE: EUC_2D`, including the ten files in `TSP Challenge/`, and accepts compatible `.tsp` uploads. Distances follow TSPLIB's nearest-integer EUC_2D rule. Hill-climbing methods accept only shortening moves; Tabu Search can accept temporary worsening moves and returns its best-so-far tour. Results are heuristic and are not guaranteed to be optimal. The starting city is not a control because changing where a closed tour is written from does not change its length.

## Run locally

Use Python 3.10 or newer:

```powershell
python -m pip install -r requirements.txt
streamlit run "Traveling Salesman Challenge.py"
```

The bundled dataset directory is resolved relative to the application file, so the launch working directory does not matter.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Deploy to Streamlit Community Cloud

1. Push this project to a GitHub repository, including `Traveling Salesman Challenge.py`, `tsp_solver.py`, `requirements.txt`, and the `TSP Challenge/` directory.
2. In Streamlit Community Cloud, create an app from that repository and select `Traveling Salesman Challenge.py` as the main file.
3. Deploy. Community Cloud installs the packages listed in `requirements.txt`; keep the bundled `.tsp` files in the repository so they are available to the app.