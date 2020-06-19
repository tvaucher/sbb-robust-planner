import datetime
import math
import os
from typing import Tuple

import pandas as pd

from hdfs3 import HDFileSystem

DEFAULT_DELAY = 87.8  # Precomputed avg delay

TRAIN_DICT = {
    "S-Bahn": "S",
    "Intercity": "IC",
    "Eurocity": "EC",
    "InterRegio": "IR",
    "RegioExpress": "RE",
    "RegionalZug": "S",
    "Eurostar": "EC",
    "TGV": "TGV",
    "ICE": "ICE",
}


def get_path(p: str, prefix: str = "/user/jjgweber") -> str:
    """ get HDFS path of the dataframe `p` """
    return os.path.join(prefix, p)


hdfs = HDFileSystem(
    host="hdfs://iccluster044.iccluster.epfl.ch", port=8020, user="ebouille"
)


def load_parquet_from_hdfs(p: str) -> pd.DataFrame:
    """ Load the dataframe `p` from HDFS  """
    df = pd.DataFrame()
    for path in hdfs.ls(get_path(p)):
        if "SUCCESS" not in path:
            with hdfs.open(path) as f:
                df = df.append(pd.read_parquet(f))
    print(p, "loaded: ", len(df), "lines")
    df.to_parquet(os.path.join("../data", p + ".parquet.gz"))
    return df


def to_parquet():
    """ Transform all the dataframes from HDFS to local """
    for p in [
        "bus_delays",
        "bus_delays_without_time",
        "tram_delays",
        "train_delays",
        "train_delays_without_type",
    ]:
        load_parquet_from_hdfs(p)


def load_parquet(p: str) -> pd.DataFrame:
    """ Load dataframe `p` from local path """
    return pd.read_parquet(get_path(p + ".parquet.gz", prefix="../data"))


class Delay:
    def __init__(self):
        """ Create the Delay utilitary by loading the necessary dataframes """
        self.bus_delays = load_parquet("bus_delays")
        self.bus_delays_without_time = load_parquet("bus_delays_without_time")
        self.tram_delays = load_parquet("tram_delays")
        self.train_delays = load_parquet("train_delays")
        self.train_delays_without_type = load_parquet("train_delays_without_type")

    def q_stop(
        self,
        stop_name: str,
        arrival_time: datetime.datetime,
        next_departure_time: datetime.datetime,
        transport_type: str,
    ) -> Tuple[float, float]:
        """
        Get the probability of catching the connection leaving at `next_departure_time`
        at a given station for a given transport type and arrival time

        Args:
            stop_name (str): Station full name
            arrival_time (datetime.datetime): Arrival time at station
            next_departure_time (datetime.datetime): Departure time at station
            transport_type (str): Type of the transport (`route_desc`)

        Returns:
            Tuple[float, float]: a tuple probability of catching connection, average delay at this station for this transport type
        """
        delay = (next_departure_time - arrival_time).total_seconds()

        if arrival_time.time() < datetime.time(9, 30, 0):
            timeframe = "morning"
        elif arrival_time.time() < datetime.time(16, 0, 0):
            timeframe = "day"
        else:
            timeframe = "evening"

        # default value in case we have no historical data
        default = DEFAULT_DELAY

        # in case of bus
        if transport_type == "Bus":
            delay_row = self.bus_delays[
                (self.bus_delays.stop_name == stop_name)
                & (self.bus_delays.timeframe == timeframe)
            ]

            if delay_row.empty:
                avg = default

            # if we have more than 300 values, we use the time parameter
            elif delay_row.n_rows.values[0] > 300:
                avg = delay_row.avg_delay.values[0]

            # else we only take into account the stop
            else:
                avg = self.bus_delays_without_time[
                    self.bus_delays_without_time.stop_name == stop_name
                ]

        # in case of tram (we always have > 400 historical values)
        elif transport_type == "Tram":
            delay_row = self.tram_delays[
                (self.tram_delays.stop_name == stop_name)
                & (self.tram_delays.timeframe == timeframe)
            ]

            avg = delay_row.avg_delay.values[0] if not delay_row.empty else default

        # in case of train
        else:
            train_type = TRAIN_DICT.get(transport_type, "General")

            # special cases for trains such as IR/IC/TGV/...
            if train_type != "General":
                delay_row = self.train_delays[
                    (self.train_delays.stop_name == stop_name)
                    & (self.train_delays.timeframe == timeframe)
                    & (self.train_delays.train_type2 == train_type)
                ]

                if delay_row.empty:
                    avg = default

                # if we have more than 300 values, we use the train_type parameter
                elif delay_row.n_rows.values[0] > 300:
                    avg = delay_row.avg_delay.values[0]

                # otherwise only take stop and time into account
                else:
                    avg = self.train_delays_without_type[
                        (self.train_delays_without_type.stop_name == stop_name)
                        & (self.train_delays_without_type.timeframe == timeframe)
                    ].avg_delay.values[0]

            else:
                delay_row = self.train_delays_without_type[
                    (self.train_delays_without_type.stop_name == stop_name)
                    & (self.train_delays_without_type.timeframe == timeframe)
                ]

                avg = delay_row.avg_delay.values[0] if not delay_row.empty else default

        return 1 - math.exp(-1 / avg * delay), avg
