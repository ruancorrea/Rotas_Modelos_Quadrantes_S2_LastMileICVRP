from haversine import haversine

from loggibud.v1.types import Delivery, Point
from typing import Optional, List, Dict


def dist_route_delivery(e: List[Delivery], rota_modelo_deliveries: List[Delivery]) -> float:
    sum = 0
    for x in e:
        sum += min(haversine((x.point.lat, x.point.lng), (d.point.lat, d.point.lng))
                    for d in rota_modelo_deliveries)
    return sum
    

def dist(local: Point, i: Delivery) -> float:
    return haversine((local.lat, local.lng), (i.point.lat, i.point.lng))

