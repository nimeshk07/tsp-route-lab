"""TSPLIB EUC_2D parsing and a from-scratch TSP heuristic."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, hypot
from pathlib import Path
from time import perf_counter


@dataclass(frozen=True)
class City:
    node_id: int
    x: float
    y: float


@dataclass(frozen=True)
class TSPInstance:
    name: str
    cities: tuple[City, ...]


@dataclass(frozen=True)
class TourResult:
    tour: tuple[int, ...]
    initial_length: int
    length: int
    elapsed_seconds: float
    starts_tried: int
    improvement_moves: int
    construction_method: str
    improvement_method: str


def parse_tsplib_text(text: str, source_name: str = "Uploaded instance") -> TSPInstance:
    """Parse a TSPLIB coordinate instance using TYPE TSP and EUC_2D costs."""
    headers: dict[str, str] = {}
    cities: list[City] = []
    in_coordinates = False
    reached_eof = False

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        upper_line = line.upper()
        if upper_line == "NODE_COORD_SECTION":
            if in_coordinates:
                raise ValueError(f"{source_name}:{line_number}: duplicate NODE_COORD_SECTION")
            in_coordinates = True
            continue
        if upper_line == "EOF":
            reached_eof = True
            break
        if not in_coordinates:
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().upper()] = value.strip()
            elif len(line.split()) >= 2:
                key, value = line.split(None, 1)
                headers[key.strip().upper()] = value.strip()
            continue

        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"{source_name}:{line_number}: expected node ID and two coordinates")
        try:
            node_id = int(fields[0])
            x = float(fields[1])
            y = float(fields[2])
        except ValueError as error:
            raise ValueError(f"{source_name}:{line_number}: invalid node ID or coordinate") from error
        cities.append(City(node_id=node_id, x=x, y=y))

    if not in_coordinates:
        raise ValueError(f"{source_name}: missing NODE_COORD_SECTION")
    if not reached_eof:
        raise ValueError(f"{source_name}: missing EOF marker")
    if headers.get("TYPE", "").upper() != "TSP":
        raise ValueError(f"{source_name}: only TYPE: TSP is supported")
    if headers.get("EDGE_WEIGHT_TYPE", "").upper() != "EUC_2D":
        raise ValueError(f"{source_name}: only EDGE_WEIGHT_TYPE: EUC_2D is supported")
    try:
        dimension = int(headers["DIMENSION"])
    except (KeyError, ValueError) as error:
        raise ValueError(f"{source_name}: a valid DIMENSION header is required") from error
    if dimension < 2:
        raise ValueError(f"{source_name}: DIMENSION must be at least 2")
    if len(cities) != dimension:
        raise ValueError(
            f"{source_name}: DIMENSION is {dimension}, but found {len(cities)} coordinate rows"
        )
    node_ids = [city.node_id for city in cities]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError(f"{source_name}: node IDs must be unique")
    if any(not (-float("inf") < value < float("inf")) for city in cities for value in (city.x, city.y)):
        raise ValueError(f"{source_name}: coordinates must be finite numbers")

    name = headers.get("NAME") or Path(source_name).stem or "TSP instance"
    return TSPInstance(name=name, cities=tuple(cities))


def load_tsplib_file(path: str | Path) -> TSPInstance:
    file_path = Path(path)
    return parse_tsplib_text(file_path.read_text(encoding="utf-8-sig"), file_path.name)


def euclidean_distance(city_a: City, city_b: City) -> int:
    """Return TSPLIB EUC_2D distance, rounded to the nearest integer."""
    return floor(hypot(city_a.x - city_b.x, city_a.y - city_b.y) + 0.5)


def tour_length(tour: tuple[int, ...] | list[int], distances: list[list[int]]) -> int:
    return sum(distances[tour[index]][tour[(index + 1) % len(tour)]] for index in range(len(tour)))


def _nearest_neighbor_tour(start_node: int, distances: list[list[int]]) -> list[int]:
    unvisited = set(range(len(distances)))
    unvisited.remove(start_node)
    tour = [start_node]
    while unvisited:
        current = tour[-1]
        next_city = min(unvisited, key=lambda candidate: (distances[current][candidate], candidate))
        tour.append(next_city)
        unvisited.remove(next_city)
    return tour


def _cheapest_insertion_tour(distances: list[list[int]]) -> list[int]:
    count = len(distances)
    if count == 2:
        return [0, 1]

    first, second = 0, 1
    farthest_distance = -1
    for left in range(count - 1):
        for right in range(left + 1, count):
            if distances[left][right] > farthest_distance:
                first, second = left, right
                farthest_distance = distances[left][right]

    third = max(
        (city for city in range(count) if city not in (first, second)),
        key=lambda city: (min(distances[city][first], distances[city][second]), -city),
    )
    tour = [first, third, second]
    remaining = set(range(count)) - set(tour)
    while remaining:
        best_increase: int | None = None
        best_city = -1
        best_position = -1
        for city in sorted(remaining):
            for position in range(len(tour)):
                before = tour[position]
                after = tour[(position + 1) % len(tour)]
                increase = distances[before][city] + distances[city][after] - distances[before][after]
                if best_increase is None or (increase, city, position) < (
                    best_increase,
                    best_city,
                    best_position,
                ):
                    best_increase, best_city, best_position = increase, city, position
        tour.insert(best_position + 1, best_city)
        remaining.remove(best_city)
    return tour


def _edge_pair_key(
    first_edge: tuple[int, int], second_edge: tuple[int, int]
) -> tuple[object, ...]:
    normalized_edges = tuple(sorted((tuple(sorted(first_edge)), tuple(sorted(second_edge)))))
    return ("2-opt", *normalized_edges)


def _best_neighborhood_move(
    tour: list[int],
    distances: list[list[int]],
    methods: tuple[str, ...],
    tabu_expirations: dict[tuple[object, ...], int] | None = None,
    candidate_neighbors: list[list[int]] | None = None,
    iteration: int = 0,
    allow_worse: bool = False,
    first_improvement: bool = False,
    current_length: int = 0,
    best_length: int = 0,
) -> tuple[int, str, int, int, tuple[object, ...], tuple[object, ...]] | None:
    count = len(tour)
    best_move = None

    def consider(
        delta: int,
        method: str,
        first: int,
        second: int,
        move_key: tuple[object, ...],
        inverse_key: tuple[object, ...],
    ) -> None:
        nonlocal best_move
        if not allow_worse and delta >= 0:
            return
        is_tabu = tabu_expirations is not None and tabu_expirations.get(move_key, -1) > iteration
        if is_tabu and current_length + delta >= best_length:
            return
        candidate = (delta, method, first, second, move_key, inverse_key)
        if best_move is None or candidate[:4] < best_move[:4]:
            best_move = candidate

    if "2-opt" in methods:
        positions = {city: position for position, city in enumerate(tour)}
        for first in range(count - 1):
            first_next = (first + 1) % count
            if candidate_neighbors is None:
                last_positions = range(first + 2, count)
            else:
                candidate_positions = set()
                for neighbor in candidate_neighbors[tour[first]]:
                    candidate_positions.add(positions[neighbor])
                for neighbor in candidate_neighbors[tour[first_next]]:
                    candidate_positions.add(positions[neighbor] - 1)
                last_positions = sorted(
                    position for position in candidate_positions if first + 2 <= position < count
                )
            for last in last_positions:
                if first == 0 and last == count - 1:
                    continue
                last_next = (last + 1) % count
                old_first = (tour[first], tour[first_next])
                old_second = (tour[last], tour[last_next])
                new_first = (tour[first], tour[last])
                new_second = (tour[first_next], tour[last_next])
                delta = (
                    distances[new_first[0]][new_first[1]]
                    + distances[new_second[0]][new_second[1]]
                    - distances[old_first[0]][old_first[1]]
                    - distances[old_second[0]][old_second[1]]
                )
                consider(
                    delta,
                    "2-opt",
                    first,
                    last,
                    _edge_pair_key(old_first, old_second),
                    _edge_pair_key(new_first, new_second),
                )
                if first_improvement and best_move is not None:
                    break
            if first_improvement and best_move is not None:
                break

    if "Swap" in methods:
        positions = {city: position for position, city in enumerate(tour)}
        for first in range(count - 1):
            if candidate_neighbors is None:
                second_positions = range(first + 1, count)
            else:
                second_positions = sorted(
                    position
                    for city in candidate_neighbors[tour[first]]
                    if (position := positions[city]) > first
                )
            for second in second_positions:
                affected_edges = {
                    (first - 1) % count,
                    first,
                    (second - 1) % count,
                    second,
                }
                old_cost = sum(
                    distances[tour[edge]][tour[(edge + 1) % count]] for edge in affected_edges
                )
                new_cost = 0
                for edge in affected_edges:
                    left = tour[second] if edge == first else tour[first] if edge == second else tour[edge]
                    right_index = (edge + 1) % count
                    right = (
                        tour[second]
                        if right_index == first
                        else tour[first]
                        if right_index == second
                        else tour[right_index]
                    )
                    new_cost += distances[left][right]
                city_pair = tuple(sorted((tour[first], tour[second])))
                key = ("swap", *city_pair)
                consider(new_cost - old_cost, "Swap", first, second, key, key)
                if first_improvement and best_move is not None:
                    break
            if first_improvement and best_move is not None:
                break

    if "Relocate" in methods:
        reduced_count = count - 1
        positions = {city: position for position, city in enumerate(tour)}
        for first in range(count):
            city = tour[first]
            previous = tour[(first - 1) % count]
            following = tour[(first + 1) % count]
            if candidate_neighbors is None:
                insertion_positions = range(reduced_count)
            else:
                insertion_positions = sorted(
                    {
                        (positions[neighbor] - (positions[neighbor] > first) + 1) % reduced_count
                        for neighbor in candidate_neighbors[city]
                        if neighbor != city
                    }
                )
            for insertion_position in insertion_positions:
                before_index = (insertion_position - 1) % reduced_count
                after_index = insertion_position
                before = tour[before_index if before_index < first else before_index + 1]
                after = tour[after_index if after_index < first else after_index + 1]
                if before == city or after == city or before == previous:
                    continue
                delta = (
                    distances[previous][following]
                    - distances[previous][city]
                    - distances[city][following]
                    + distances[before][city]
                    + distances[city][after]
                    - distances[before][after]
                )
                move_key = ("relocate", city, before)
                inverse_key = ("relocate", city, previous)
                consider(
                    delta,
                    "Relocate",
                    first,
                    insertion_position,
                    move_key,
                    inverse_key,
                )
                if first_improvement and best_move is not None:
                    break
            if first_improvement and best_move is not None:
                break
    return best_move


def _apply_move(tour: list[int], move: tuple[int, str, int, int, tuple[object, ...], tuple[object, ...]]) -> None:
    _, method, first, second, _, _ = move
    if method == "2-opt":
        tour[first + 1 : second + 1] = reversed(tour[first + 1 : second + 1])
    elif method == "Swap":
        tour[first], tour[second] = tour[second], tour[first]
    else:
        city = tour.pop(first)
        tour.insert(second, city)


def solve_tsp(
    instance: TSPInstance,
    construction_method: str = "Nearest Neighbor",
    improvement_method: str = "Mixed neighborhoods",
    starts: int = 8,
    max_improvement_moves: int = 100,
    tabu_tenure: int = 12,
) -> TourResult:
    """Construct a tour and improve it with a selectable local-search method."""
    if starts < 1:
        raise ValueError("starts must be at least 1")
    if max_improvement_moves < 0:
        raise ValueError("max_improvement_moves cannot be negative")
    if tabu_tenure < 1:
        raise ValueError("tabu_tenure must be at least 1")
    if construction_method not in ("Nearest Neighbor", "Cheapest Insertion"):
        raise ValueError(f"Unsupported construction method: {construction_method}")
    supported_methods = ("2-opt", "Swap", "Relocate", "Mixed neighborhoods", "Tabu Search")
    if improvement_method not in supported_methods:
        raise ValueError(f"Unsupported improvement method: {improvement_method}")

    started_at = perf_counter()
    cities = instance.cities
    count = len(cities)
    distances = [
        [euclidean_distance(cities[first], cities[second]) for second in range(count)]
        for first in range(count)
    ]
    if construction_method == "Nearest Neighbor":
        start_count = min(starts, count)
        start_nodes = [index * count // start_count for index in range(start_count)]
        starting_tours = [_nearest_neighbor_tour(start_node, distances) for start_node in start_nodes]
    else:
        start_count = 1
        starting_tours = [_cheapest_insertion_tour(distances)]
    best_tour: list[int] | None = None
    best_length: int | None = None
    for tour in starting_tours:
        length = tour_length(tour, distances)
        if best_length is None or length < best_length:
            best_tour, best_length = tour, length

    assert best_tour is not None and best_length is not None
    initial_length = best_length
    current_tour = best_tour.copy()
    current_length = best_length
    tabu_expirations: dict[tuple[object, ...], int] = {}
    if improvement_method == "2-opt":
        neighborhoods = ("2-opt",)
    elif improvement_method == "Swap":
        neighborhoods = ("Swap",)
    elif improvement_method == "Relocate":
        neighborhoods = ("Relocate",)
    else:
        neighborhoods = ("2-opt", "Swap", "Relocate")

    accepted_moves = 0
    candidate_neighbors = None
    if improvement_method in ("Mixed neighborhoods", "Tabu Search"):
        candidate_limit = min(16, count - 1)
        candidate_neighbors = [
            sorted(
                (other for other in range(count) if other != city),
                key=lambda other: (distances[city][other], other),
            )[:candidate_limit]
            for city in range(count)
        ]
    for iteration in range(max_improvement_moves):
        use_tabu = improvement_method == "Tabu Search"
        if use_tabu:
            move = _best_neighborhood_move(
                current_tour,
                distances,
                neighborhoods,
                tabu_expirations,
                candidate_neighbors,
                iteration,
                allow_worse=True,
                current_length=current_length,
                best_length=best_length,
            )
        else:
            neighborhood_moves = [
                move
                for neighborhood in neighborhoods
                if (
                    move := _best_neighborhood_move(
                        current_tour,
                        distances,
                        (neighborhood,),
                        candidate_neighbors=candidate_neighbors,
                        first_improvement=True,
                    )
                ) is not None
            ]
            move = min(neighborhood_moves, key=lambda candidate: candidate[0]) if neighborhood_moves else None
        if move is None:
            break
        delta, _, _, _, _, inverse_key = move
        _apply_move(current_tour, move)
        current_length += delta
        accepted_moves += 1
        if use_tabu:
            tabu_expirations[inverse_key] = iteration + tabu_tenure + 1
        if current_length < best_length:
            best_tour = current_tour.copy()
            best_length = current_length

    final_length = tour_length(best_tour, distances)
    return TourResult(
        tour=tuple(best_tour),
        initial_length=initial_length,
        length=final_length,
        elapsed_seconds=perf_counter() - started_at,
        starts_tried=start_count,
        improvement_moves=accepted_moves,
        construction_method=construction_method,
        improvement_method=improvement_method,
    )