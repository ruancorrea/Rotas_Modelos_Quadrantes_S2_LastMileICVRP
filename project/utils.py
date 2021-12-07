from typing import Optional, List


from loggibud.v1.types import (
    Delivery,
    CVRPInstance,
    CVRPSolution,
    CVRPSolutionVehicle,
    JSONDataclassMixin,
    Point,
)

from dataclasses import dataclass
from sklearn.cluster import KMeans

from loggibud.v1.baselines.shared.ortools import (
    solve as ortools_solve,
    ORToolsParams,
)


@dataclass
class AlocParams(JSONDataclassMixin):
    seed: int = 0
    ortools_tsp_params: Optional[ORToolsParams] = ORToolsParams(
                max_vehicles=1,
                time_limit_ms=1_000,
            )
    pts_batch: Optional[int] = 1000
    nivel_s2cells: Optional[int] = 7
    num_ucs: Optional[int] = 28

    @classmethod
    def get_baseline(cls):
        return cls(
            ortools_tsp_params=ORToolsParams(
                max_vehicles=1,
                time_limit_ms=1_000,
            ),
        )


@dataclass
class AlocModel:
    params: AlocParams
    clustering: KMeans


@dataclass
class UCModel:
    C: int
    phi: List[Delivery]
    deliveries: List[Delivery]



def createUC() -> UCModel:
    return UCModel(
        C= 0,
        phi= [],
        deliveries= []
    )


def create_instanceCVRP(instance: CVRPInstance, deliveries: List[Delivery], name: str, x: int):
    return CVRPInstance(
                name = name,
                region= "",
                origin= instance.origin,
                vehicle_capacity = x * instance.vehicle_capacity,
                deliveries= deliveries
            )

def batchInstance(instance: CVRPInstance, n: int, num_points: int):
    batch_points = []
    k = 0
    while k < num_points and n < len(instance.deliveries): ## separando a fatia da instancia -> BATCH
        batch_points.append(instance.deliveries[n])
        k += 1
        n += 1

    return n, batch_points



dictOffilineDF0 = {"cvrp-0-df-90": 1719.28162,
                    "cvrp-0-df-91": 1493.65650,
                    "cvrp-0-df-92": 1614.18900,
                    "cvrp-0-df-93": 1371.36930,
                    "cvrp-0-df-94": 1658.22240,
                    "cvrp-0-df-95": 1921.84922,
                    "cvrp-0-df-96": 1797.01350,
                    "cvrp-0-df-97": 1513.69562,
                    "cvrp-0-df-98": 1505.49190,
                    "cvrp-0-df-99": 1477.50658,
                    "cvrp-0-df-100": 1934.26072,
                    "cvrp-0-df-101": 1799.57220,
                    "cvrp-0-df-102": 1877.56800,
                    "cvrp-0-df-103": 1945.03250,
                    "cvrp-0-df-104": 1803.16120,
                    "cvrp-0-df-105": 1520.84120,
                    "cvrp-0-df-106": 1845.14270,
                    "cvrp-0-df-107": 2039.06770,
                    "cvrp-0-df-108": 1866.76710,
                    "cvrp-0-df-109": 1861.27540,
                    "cvrp-0-df-110": 2024.93866,
                    "cvrp-0-df-111": 1766.13040,
                    "cvrp-0-df-112": 1811.79880,
                    "cvrp-0-df-113": 1608.11110,
                    "cvrp-0-df-114": 1872.92870,
                    "cvrp-0-df-115": 1791.09120,
                    "cvrp-0-df-116": 1634.39928,
                    "cvrp-0-df-117": 1484.92774,
                    "cvrp-0-df-118": 1657.33882,
                    "cvrp-0-df-119": 1523.94120
                    }

dictOffilinePA0 = {
                    "cvrp-0-pa-90": 607.69230,
                    "cvrp-0-pa-91": 605.58880,
                    "cvrp-0-pa-92": 627.20640,
                    "cvrp-0-pa-93": 534.85710,
                    "cvrp-0-pa-94": 641.61870,
                    "cvrp-0-pa-95": 800.52120,
                    "cvrp-0-pa-96": 696.28118,
                    "cvrp-0-pa-97": 609.68670,
                    "cvrp-0-pa-98": 635.04320,
                    "cvrp-0-pa-99": 941.28450,
                    "cvrp-0-pa-100": 633.63830,
                    "cvrp-0-pa-101": 625.41242,
                    "cvrp-0-pa-102": 655.13880,
                    "cvrp-0-pa-103": 590.82290,
                    "cvrp-0-pa-104": 758.43430,
                    "cvrp-0-pa-105": 764.12520,
                    "cvrp-0-pa-106": 752.08480,
                    "cvrp-0-pa-107": 609.12526,
                    "cvrp-0-pa-108": 643.56936,
                    "cvrp-0-pa-109": 578.25260,
                    "cvrp-0-pa-110": 633.41986,
                    "cvrp-0-pa-111": 580.42140,
                    "cvrp-0-pa-112": 755.66070,
                    "cvrp-0-pa-113": 585.28940,
                    "cvrp-0-pa-114": 668.62870,
                    "cvrp-0-pa-115": 678.99704,
                    "cvrp-0-pa-116": 634.25884,
                    "cvrp-0-pa-117": 682.54110,
                    "cvrp-0-pa-118": 678.79628,
                    "cvrp-0-pa-119": 665.91168
            }


dictOffilineRJ0 = {
                "cvrp-0-rj-90": 4862.00618,
                "cvrp-0-rj-91": 5158.17758,
                "cvrp-0-rj-92": 5098.99614,
                "cvrp-0-rj-93": 4967.33268,
                "cvrp-0-rj-94": 4687.93670,
                "cvrp-0-rj-95": 4172.66190,
                "cvrp-0-rj-96": 4744.18818,
                "cvrp-0-rj-97": 4322.57488,
                "cvrp-0-rj-98": 4999.90870,
                "cvrp-0-rj-99": 5262.07426,
                "cvrp-0-rj-100": 4814.98758,
                "cvrp-0-rj-101": 4256.30204,
                "cvrp-0-rj-102": 4367.56994,
                "cvrp-0-rj-103": 5331.12676,
                "cvrp-0-rj-104": 4045.14080,
                "cvrp-0-rj-105": 5221.41196,
                "cvrp-0-rj-106": 4777.87994,
                "cvrp-0-rj-107": 4416.07146,
                "cvrp-0-rj-108": 4448.93778,
                "cvrp-0-rj-109": 4580.32076,
                "cvrp-0-rj-110": 4342.81864,
                "cvrp-0-rj-111": 5125.50500,
                "cvrp-0-rj-112": 4849.18374,
                "cvrp-0-rj-113": 5439.31818,
                "cvrp-0-rj-114": 5139.47512,
                "cvrp-0-rj-115": 5219.27022,
                "cvrp-0-rj-116": 4592.44578,
                "cvrp-0-rj-117": 5170.00134,
                "cvrp-0-rj-118": 5369.29378,
                "cvrp-0-rj-119": 4627.40872
                }