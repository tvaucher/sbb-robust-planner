import datetime
from collections import defaultdict
from functools import reduce
from itertools import combinations
from operator import itemgetter
from typing import DefaultDict, List, Optional, Tuple, Union

import networkx as nx
import pandas as pd

from .delay import Delay
from .utils import (RouteNode, get_number_trips, parse_timestamp,
                    print_timestamp, same_departure_time)

CHANGE_TRAIN_TIME = 120  # seconds
WAITING_TRESHOLD = 30 * 60  # seconds, maximum waiting time at a station


class RoutePlanner:
    def __init__(self, graph, start, destination, arr_time):
        self.graph = graph
        self.start = start
        self.destination = destination
        self.arr_time = (
            parse_timestamp(arr_time)
            if len(arr_time) > 5
            else parse_timestamp(arr_time, format_="%H:%M")
        )
        self.route = defaultdict(RouteNode)
        self.blocked_trips = set()
        self.blocked_trips_old = set()
        self.max_iter = 19

    def __compute_weight(self, source: str, target: str, st_edges) -> Optional[float]:
        """
        Compute the weight for Dijkstra between source and target. As G is a MultiDiGraph, we choose the best edge to follow.
        This methods update the route dictionary that keep tracks of the route infos

        Args:
            source (str): Source node
            target (str): Target node
            st_edges (nx.Edges): Edges set between s and t

        Returns:
            Optional[float]: the minimum weight between s and t or None when we want to hide the node
        """
        # Init data from previously visited source node (which will never be checked again because we use Dijsktra).
        latest_route = self.route[source].get_latest()
        next_trip = latest_route["trip_id"]  # Trip to leave the source node
        next_dep_time = latest_route[
            "dep_time"
        ]  # At what time we need to leave the source node

        # Init weight for this edge
        # Idea is to minimize the duration, while maintaining a feasible path
        # i.e. want to maximize prev_dep_time (as late as possible)
        best_edge_idx = -1
        min_weight = float("inf")  # weight := travel duration + waiting time
        max_prev_dep_time = parse_timestamp("00:00:01")
        transfer = False

        # Iterate among possible edge between
        for (edge_idx, edge) in st_edges.items():
            if edge["trip_id"] in self.blocked_trips:
                continue  # Don't want to evaluate a blocked path

            # In case of a transfer
            if edge["trip_id"].startswith("TRANSFER"):
                if not next_trip.startswith(
                    "TRANSFER"
                ):  # Don't want to walk more than 500m at once
                    weight = edge["t_time"]
                    if weight < min_weight:
                        min_weight = weight
                        best_edge_idx = edge_idx
                        transfer = True
                        max_prev_dep_time = next_dep_time - datetime.timedelta(
                            seconds=weight
                        )

            # In case of a trip that isn't blocked (i.e. seen in a previous iteration of plan)
            else:
                t_time = (
                    0
                    if (
                        source
                        == self.destination  # No add transfer time at the destination
                        or edge["trip_id"] == next_trip  # We stay on the train
                        or next_trip.startswith("TRANSFER")
                    )  # The 2 minutes are already included in the trip
                    else CHANGE_TRAIN_TIME
                )
                edge["t_time"] = t_time
                # Check the feasibility of the trip
                if (
                    edge["arr_time"] + datetime.timedelta(seconds=t_time)
                    <= next_dep_time  # Assuming no delay, we can catch our train
                    and (next_dep_time - edge["arr_time"]).seconds <= WAITING_TRESHOLD
                ):  # Don't want to wait too much at a station
                    waiting_time = (next_dep_time - edge["arr_time"]).seconds
                    weight = edge["duration"] + waiting_time
                    if weight < min_weight:
                        min_weight = weight
                        best_edge_idx = edge_idx
                        transfer = False
                        max_prev_dep_time = edge["prev_dep_time"]

        if best_edge_idx >= 0:  # We found an edge that we could take
            best_edge = st_edges[best_edge_idx]
            if min_weight < self.route[target][source].get("weight", float("inf")):
                self.route[target][source] = {
                    "weight": min_weight,
                    "trip_id": best_edge["trip_id"],
                    "dep_time": max_prev_dep_time,
                    "t_time": best_edge["t_time"],
                }
                if not transfer:  # Add more trip info if it's not a transfer
                    self.route[target][source].update(
                        {
                            "trip_info": {
                                "arr_time": best_edge["arr_time"],
                                "duration": best_edge["duration"],
                                "trip_headsign": best_edge["trip_headsign"],
                                "route_desc": best_edge["route_desc"],
                                "route_short_name": best_edge["route_short_name"],
                                "stop_sequence": best_edge["stop_sequence"],
                            },
                        }
                    )
            return min_weight

        return None  # o/w hide this edge from Dijsktra's algo

    def __print_path_result(self, path: List[str], route: DefaultDict[str, RouteNode]):
        """ Helper to print some info about the current path """
        # path = list(reversed(path))
        print("***PATH***")
        print(path)
        print()
        print("***ROUTE***")
        for station1, station2 in zip(path, path[1:]):
            trip_data = route[station1][station2]
            if trip_data["trip_id"].startswith("TRANSFER"):
                print(
                    "At station",
                    station1,
                    "walk to",
                    station2,
                    "leave latest",
                    print_timestamp(trip_data["dep_time"]),
                    trip_data["t_time"],
                )
            else:
                trip_info = trip_data["trip_info"]
                print(
                    "At station",
                    station1,
                    "take",
                    trip_data["trip_id"],
                    ":",
                    trip_info["route_desc"],
                    trip_info["route_short_name"],
                    "headed to",
                    trip_info["trip_headsign"],
                    "to",
                    station2,
                    "that leaves at",
                    print_timestamp(trip_data["dep_time"]),
                    "and arrives at",
                    print_timestamp(trip_info["arr_time"]),
                    trip_data["t_time"],
                )
        print()

    def plan(
        self, info: bool = False
    ) -> Tuple[Optional[List[str]], Optional[DefaultDict[str, RouteNode]]]:
        """
        Return the shortest path using Dijkstra in the current graph

        Args:
            info (bool, optional): print additional debugging info. Defaults to False.

        Returns:
            Tuple[Optional[List[str]], Optional[DefaultDict[str, RouteNode]]]: a Tuple containing the path and route information. Tuple of Nones if no path
        """
        if nx.has_path(self.graph, self.destination, self.start):
            # Reset node data
            self.route = defaultdict(RouteNode)
            self.route[self.destination] = RouteNode(
                self.destination, {"trip_id": "DONE", "dep_time": self.arr_time}
            )
            try:
                path = nx.dijkstra_path(
                    self.graph, self.destination, self.start, self.__compute_weight
                )
            except nx.NetworkXNoPath:  # No Path
                return None, None

            return list(reversed(path)), self.route

        else:
            if info:
                print("No such path exist")
            return None, None

    def __get_next_best_path(
        self,
        current_path: List[str],
        current_route: DefaultDict[str, RouteNode],
        info: bool = False,
    ) -> Tuple[Optional[List[List[str]]], Optional[List[DefaultDict[str, RouteNode]]]]:
        """
        Get the following next best shortest path. Take the current shortest path and try to block
        trips one after the other. Returns only the fastest paths from this iteration

        Args:
            current_path (List[str]): [description]
            current_route (DefaultDict[str, RouteNode]): [description]
            info (bool, optional): [description]. Defaults to False.

        Returns:
            Tuple[Optional[List[List[str]]], Optional[List[DefaultDict[str, RouteNode]]]]: the best paths and routes for this iteration
        """
        trips = {
            current_route[station1][station2]["trip_id"]
            for station1, station2 in zip(current_path, current_path[1:])
        }
        trips |= set(combinations(trips, 2))  # at all pairs of trips
        if info:
            print("Trips to block: ", trips)
        best_path, best_route = None, None
        latest_dep_time, trip = parse_timestamp("00:00:01"), set()

        if info:
            print(
                "Trying to find new paths, currently blocked ", self.blocked_trips_old
            )
        for blocked_trip in trips:
            if (
                isinstance(blocked_trip, tuple)
                and set(blocked_trip) <= self.blocked_trips_old
            ) or (
                isinstance(blocked_trip, str) and blocked_trip in self.blocked_trips_old
            ):
                continue  # not interested in blocking already blocked path
            if info:
                print("Blocking ", blocked_trip)
            self.blocked_trips = (
                self.blocked_trips_old | {blocked_trip}
                if isinstance(blocked_trip, str)
                else self.blocked_trips_old | set(blocked_trip)
            )
            path, route = self.plan(
                info=info
            )  # Try to find a path not using previous blocked trips or a selected trip
            if path and route:
                dep_time = route[path[0]][path[1]][
                    "dep_time"
                ]  # Get plan departure time
                if dep_time > latest_dep_time:  # Found a faster trip with constraints
                    best_path, best_route = [path], [route]
                    latest_dep_time, trip = dep_time, {blocked_trip}
                elif (
                    dep_time == latest_dep_time and best_path and best_route
                ):  # Found a path that is as fast, will need to desambiguate with probabilities later if needed
                    best_path.append(path)
                    best_route.append(route)
                    trip.add(blocked_trip)

        # update blocked_trips for further iteration if needed
        trip = reduce(
            lambda x, y: x | set(y) if isinstance(y, tuple) else x | {y}, trip, set()
        )
        print("new trips to block", trip)
        self.blocked_trips_old |= trip
        print(self.blocked_trips_old)
        return best_path, best_route

    def get_path_probability(
        self,
        path: List[str],
        route: DefaultDict[str, RouteNode],
        delay: Delay,
        stops: pd.DataFrame,
    ) -> Tuple[float, List[Tuple[str, float]]]:
        """
        Compute the probability of catching all the connexions and to be reach the destination before the wished arrival time.
        Probabilities are computed for each transferring nodes and at the last station.
        No delay is arrival delay is assumed at the very beginning.
        Departure delays are assumed to be 0 (trains always leave on time)

        Args:
            path (List[str]): Path for which to compute the probability
            route (DefaultDict[str, RouteNode]): Routes info for this path
            delay (Delay): Delay object to compute the delay
            stops (pd.DataFrame): Stops information dataframe

        Returns:
            Tuple[float, List[Tuple[str, float]]]: Probability of this path being successful, a list of tuples (trip_id, proba)
        """

        def get_station_name(stop_id):
            return stops[stops.stop_id == stop_id].iloc[0]["stop_name"]

        current_trip_id = ""
        arr_time = parse_timestamp("00:00:01")
        old_station, old_i, route_desc = "", -1, ""
        total_proba, trip_id_proba = 1.0, []
        for i, (station1, station2) in enumerate(zip(path, path[1:])):
            trip_data = route[station1][station2]
            if trip_data["trip_id"].startswith(
                "TRANSFER"
            ):  # Add t_time to arr_time in case of transfer
                if i == 0:
                    continue  # Don't want to add transfer time if we begin our journey by a transfer
                arr_time += datetime.timedelta(seconds=trip_data["t_time"])

            else:  # On a trip
                trip_info = trip_data["trip_info"]
                # not(Beginning of journey or Stay on train) => Change train, need to compute proba of missing the connection
                if current_trip_id != "" and current_trip_id != trip_data["trip_id"]:
                    dep_time = trip_data["dep_time"]
                    proba, expected_delay = delay.q_stop(
                        get_station_name(old_station), arr_time, dep_time, route_desc
                    )
                    total_proba *= proba
                    trip_id_proba.append((trip_data["trip_id"], proba))

                    route[path[old_i]][path[old_i + 1]]["trip_info"]["delay"] = (
                        expected_delay,
                        proba,
                    )  # Add expected delay on previous arrival

                current_trip_id = trip_data["trip_id"]
                arr_time = trip_info["arr_time"] + datetime.timedelta(
                    seconds=trip_data["t_time"]
                )  # update to next station
                old_station, old_i = station2, i
                route_desc = trip_info["route_desc"]

        # Finally add probability at last stop
        proba, expected_delay = delay.q_stop(
            get_station_name(old_station), arr_time, self.arr_time, route_desc
        )
        total_proba *= proba
        trip_id_proba.append((current_trip_id, proba))

        route[path[old_i]][path[old_i + 1]]["trip_info"]["delay"] = (
            expected_delay,
            proba,
        )  # Add expected delay on previous arrival

        return total_proba, trip_id_proba

    def __get_best_route(
        self,
        pathes: List[List[str]],
        routes: List[DefaultDict[str, RouteNode]],
        delay: Delay,
        stops: pd.DataFrame,
        info: bool = False,
    ) -> Tuple[List[str], DefaultDict[str, RouteNode], float]:
        """
        Get the best route among a choice of same departure time routes.
        Ordering: Highest success probability, least number of trips, least number of walking transfer

        Args:
            pathes (List[List[str]]): List of paths to evaluate
            routes (List[DefaultDict[str, RouteNode]]): Corresponding routes
            delay (Delay): Delay object
            stops (pd.DataFrame): Stops information dataframe
            info (bool, optional): print some extra info. Defaults to False.

        Returns:
            Tuple[List[str], DefaultDict[str, RouteNode], float]: best path, route and associated probability
        """
        # maximize proba and then minimize length of trip
        ordered_pathes = sorted(
            enumerate(
                (
                    -self.get_path_probability(cur_path, cur_route, delay, stops)[0],
                    *get_number_trips(cur_path, cur_route),
                )
                for cur_path, cur_route in zip(pathes, routes)
            ),
            key=itemgetter(1),
        )
        if info:
            print(ordered_pathes)
        best_path_idx, (best_proba, nb_trips, nb_walking_trips) = ordered_pathes[0]
        best_proba *= -1  # Add to take - for sorting
        return pathes[best_path_idx], routes[best_path_idx], best_proba

    def __get_next_path_by_proba(
        self,
        current_path: List[str],
        current_route: DefaultDict[str, RouteNode],
        trip_id_proba: List[Tuple[str, float]],
        threshold: float,
        delay: Delay,
        stops: pd.DataFrame,
        info: bool = False,
    ) -> Tuple[
        Optional[List[str]],
        Optional[DefaultDict[str, RouteNode]],
        Optional[List[Tuple[str, float]]],
        float,
    ]:
        """
        Instead of blocking paths at random, this method tries to incrementally block the trips with highest probability of failure
        This showed a huge improvement compared to random. To improve the quality, it also tries to block walking transfers
        as we have empiracally seen that sometimes, Dijkstra liked to take edges that make it leave the station asap.

        Args:
            current_path (List[str]): The current best path
            current_route (DefaultDict[str, RouteNode]): Current best route
            trip_id_proba (List[Tuple[str, float]]): Associated trip delays
            threshold (float): Target probability threshold
            delay (Delay): Delay object
            stops (pd.DataFrame): Stops info df
            info (bool, optional): Print additional info. Defaults to False.

        Returns:
            Tuple[Optional[List[str]], Optional[DefaultDict[str, RouteNode]], Optional[List[Tuple[str, float]]], float]:
                Best path, route, associated trip probabilites, best overall probability or None, None, None, -1 if no trips can be found
        """
        trip_id_proba = sorted(trip_id_proba, key=itemgetter(1))
        trip_stats = []
        transfers_trip = [
            current_route[station1][station2]["trip_id"]
            for station1, station2 in zip(current_path, current_path[1:])
            if current_route[station1][station2]["trip_id"].startswith("TRANSFER")
        ]
        trips = [
            frozenset(map(itemgetter(0), trip_id_proba[: i + 1]))
            for i in range(len(trip_id_proba))
        ]
        if info:
            print(
                "Trying to find new paths, currently blocked ", self.blocked_trips_old
            )
        for blocked_trip in trips + transfers_trip:
            if (
                isinstance(blocked_trip, tuple)
                and set(blocked_trip) <= self.blocked_trips_old
            ) or (
                isinstance(blocked_trip, str) and blocked_trip in self.blocked_trips_old
            ):
                continue  # not interested in blocking already blocked path
            if info:
                print("Blocking ", blocked_trip)
            self.blocked_trips = (
                self.blocked_trips_old | {blocked_trip}
                if isinstance(blocked_trip, str)
                else self.blocked_trips_old | set(blocked_trip)
            )
            path, route = self.plan(
                info=info
            )  # Try to find a path not using previous blocked trips or a selected trip
            if path and route:
                dep_time = route[path[0]][path[1]][
                    "dep_time"
                ]  # Get plan departure time
                proba, new_trip_id_proba = self.get_path_probability(
                    path, route, delay, stops
                )
                transfers, walking_transfer = get_number_trips(path, route)
                trip_stats.append(
                    {
                        "path": path,
                        "route": route,
                        "trip_id_proba": new_trip_id_proba,
                        "dep_time": dep_time,
                        "proba": proba,
                        "transfers": -transfers,
                        "walking_transfers": -walking_transfer,
                    }
                )
        if len(trip_stats) == 0:
            return None, None, None, 0.0
        filtered_trip_stats = [v for v in trip_stats if v["proba"] >= threshold]
        output_getter = itemgetter("path", "route", "trip_id_proba", "proba")
        if len(filtered_trip_stats) == 0:
            self.blocked_trips_old |= set(map(itemgetter(0), trip_id_proba))
            return output_getter(trip_stats[0])
        sorted_trip_stats = sorted(
            filtered_trip_stats,
            key=itemgetter("dep_time", "proba", "transfers", "walking_transfers"),
            reverse=True,
        )
        print(
            [
                itemgetter("dep_time", "proba", "transfers", "walking_transfers")(x)
                for x in sorted_trip_stats
            ]
        )
        return output_getter(sorted_trip_stats[0])

    def __path_arrive_earlier(
        self, cur_dep_time: datetime.datetime, increment: int = 5, max_iter: int = 5
    ) -> Tuple[Optional[List[List[str]]], Optional[List[DefaultDict[str, RouteNode]]]]:
        """
        Try to get paths that arrive earlier at the destination in order to sometimes avoid unecessary walking.
        Return new paths only if they are as fast as the current best path.

        Args:
            cur_dep_time (datetime.datetime): Current best path departure time
            increment (int, optional): Value of the increment to look for, in minutes. Defaults to 5.
            max_iter (int, optional): Number of times we want to try to increment in the past. Defaults to 5.

        Returns:
            Tuple[Optional[List[List[str]]], Optional[List[DefaultDict[str, RouteNode]]]]: New paths and routes if any are found.
        """
        pathes, routes = [], []
        for i in range(max_iter):
            new_arr_time = print_timestamp(
                self.arr_time - datetime.timedelta(minutes=increment * (i + 1))
            )
            planner = RoutePlanner(
                self.graph, self.start, self.destination, new_arr_time,
            )  # Try to find something that arrives earlier
            path, route = planner.plan()
            if path and route:
                dep_time = route[path[0]][path[1]]["dep_time"]
                print(
                    "Found a path that arrives for",
                    new_arr_time,
                    "and leaves at",
                    print_timestamp(dep_time),
                    " For reference",
                    print_timestamp(cur_dep_time),
                )
                if cur_dep_time == dep_time:  # If same departure time
                    pathes.append(path)
                    routes.append(route)
        if len(pathes) < 1:  # Found nothing interesting
            return None, None
        return pathes, routes

    def plan_robust(
        self,
        threshold: float,
        delay: Delay,
        stops: pd.DataFrame,
        info: bool = False,
        print_path: bool = False,
    ) -> Tuple[
        Optional[List[str]], Optional[DefaultDict[str, RouteNode]], Union[float, int]
    ]:
        """
        Main method to call for planning a route. Compute the fastest path to arrive at a given time and a given probability

        Args:
            threshold (float): The minimum probability to be able to have a successful journey
            delay (Delay): Delay Object
            stops (pd.DataFrame): Stops information dataframe
            info (bool, optional): print some extra info. Defaults to False.
            print_path (bool, optional): prints output path. Defaults to False.

        Returns:
            Tuple[ Optional[List[str]], Optional[DefaultDict[str, RouteNode]], Union[float, int] ]: The best path, route and probability on success.
                Otherwise returns the most robust path, route and associated probability that we found.
        """
        path, route = self.plan(info=info)
        if not path or not route:
            if print_path:
                print("No path found")
            return None, None, -1
        proba, trip_id_proba = self.get_path_probability(path, route, delay, stops)
        if proba >= threshold:
            # Want to try to improve on initial solution
            pathes, routes = self.__get_next_best_path(path, route, info)

            # Can't find something equally fast
            if (
                not pathes
                or not routes
                or not same_departure_time(path, route, pathes, routes)
            ):
                if print_path:
                    self.__print_path_result(path, route)
                return path, route, proba

            pathes.append(path)
            routes.append(route)
            path, route, proba = self.__get_best_route(
                pathes, routes, delay, stops, info
            )

            # Try to arrive earlier
            earlier_pathes, earlier_routes = self.__path_arrive_earlier(
                route[path[0]][path[1]]["dep_time"]
            )
            if not earlier_pathes or not earlier_routes:
                if print_path:
                    self.__print_path_result(path, route)
                return path, route, proba

            earlier_pathes.append(path)
            earlier_routes.append(route)
            return self.__get_best_route(
                earlier_pathes, earlier_routes, delay, stops, info
            )

        # Try to find a path that matches the wished probability threshold
        for i in range(self.max_iter):
            (
                new_path,
                new_route,
                new_trip_id_proba,
                new_proba,
            ) = self.__get_next_path_by_proba(
                path, route, trip_id_proba, threshold, delay, stops, info
            )
            if not new_path or not new_route or not new_trip_id_proba:
                return path, route, proba  # Returns current most robust path instead
            if new_proba >= threshold:
                if print_path:
                    self.__print_path_result(new_path, new_route)
                return new_path, new_route, new_proba
            trip_id_proba = new_trip_id_proba
            if proba < new_proba:  # Update current most robust path
                proba = new_proba
                path = new_path
                route = new_route

        if print_path:
            print("No path found")
        return path, route, proba  # In this case, return the most robust path we found
