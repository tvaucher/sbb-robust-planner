import datetime
from collections import defaultdict
from typing import Tuple, Optional, Any, Dict, DefaultDict


def get_number_trips(path, route) -> Tuple[int, int]:
    """ returns the number of different trips and walking transfers on this path """
    counter = 0
    counter_walking = 0
    current_trip_id = ""
    for station1, station2 in zip(path, path[1:]):
        trip_id = route[station1][station2]["trip_id"]
        if current_trip_id != trip_id:
            current_trip_id = trip_id
            counter += 1
            if trip_id.startswith("TRANSFER"):
                counter_walking += 1
    return counter, counter_walking


def same_departure_time(path, route, pathes, routes) -> bool:
    """ Check that a path and a list of candidate paths have the same departure time """
    dep_time = route[path[0]][path[1]]["dep_time"]
    new_dep_time = routes[0][pathes[0][0]][pathes[0][1]]["dep_time"]
    return dep_time == new_dep_time


def parse_timestamp(time: str, format_="%H:%M:%S") -> datetime.datetime:
    """ Parse a string timestamp to a datetime """
    return datetime.datetime.strptime(time, format_)


def print_timestamp(time: datetime.datetime, format_="%H:%M") -> str:
    """ Prepare timestamp for pretty print """
    return datetime.datetime.strftime(time, format_)


class RouteNode:
    def __init__(
        self, key: Optional[str] = None, value: Optional[Dict[str, Any]] = None
    ):
        """ Wrapper around a defaultdict that contains dictionaries of info about edges outgoing from the given Route Node """
        self._data: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)
        if key and value:
            self._data[key] = value

    def __getitem__(self, key: str) -> Dict[str, Any]:
        return self._data[key]

    def __setitem__(self, key: str, value: Any):
        self._data[key] = value

    def get_latest(self) -> Dict[str, Any]:
        """ Return the latest time we can leave this current node and still be on time """
        if len(self._data) < 1:
            raise KeyError("Calling get_latest on empty RouteNode")
        latest = parse_timestamp("00:00:01")
        best_key = ''
        for key, value in self._data.items():
            if latest < value["dep_time"]:
                latest = value["dep_time"]
                best_key = key
        return self._data[best_key]

    def __repr__(self):
        return self._data.__repr__()
