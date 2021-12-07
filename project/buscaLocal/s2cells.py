import math
from s2sphere import RegionCoverer, Cell, LatLng, LatLngRect, CellId, SphereInterval
from copy import deepcopy
import pandas as pd
from typing import Optional, List, Dict

from loggibud.v1.types import CVRPSolution, Delivery


def get_max_lat(pool: CVRPSolution) -> float: #retorna a maior latitude encontrada no cluster
    maxLat = -math.inf
    for c in pool.deliveries:
        maxLat = max(maxLat, c.point.lat)
    return maxLat


def get_max_lng(pool: CVRPSolution) -> float: #retorna a maior longitude encontrada no cluster
    maxLng = -math.inf
    for c in pool.deliveries:
        maxLng = max(maxLng, c.point.lng)
    return maxLng


def get_min_lat(pool: CVRPSolution) -> float: #retorna a menor latitude encontrada no cluster
    minLat = math.inf
    for c in pool.deliveries:
        minLat = min(minLat, c.point.lat)
    return minLat


def get_min_lng(pool: CVRPSolution) -> float: #retorna a menor longitude encontrada no cluster
    minLng = math.inf
    for c in pool.deliveries:
        minLng = min(minLng, c.point.lng)
    return minLng

# soma quadrantes da rota
def somaQdR(route: List[Delivery], cell_ids):
    quad = [0 for i in range(len(cell_ids))]
    for i in range(len(cell_ids)):
        for delivery in route.deliveries:
            p1 = LatLng.from_degrees(delivery.point.lat, delivery.point.lng)
            if (LatLngRect.from_point(p1).intersects(Cell(cell_ids[i]))):
                quad[i] += 1 
                #break

    #print(quad)
    return quad

# quadrantes das rotas
def quads_of_route(route: List[Delivery], cell_ids):
    quad = [0 for i in range(len(cell_ids))]
    for i in range(len(cell_ids)):
        for delivery in route.deliveries:
            p1 = LatLng.from_degrees(delivery.point.lat, delivery.point.lng)
            if (LatLngRect.from_point(p1).intersects(Cell(cell_ids[i]))):
                quad[i] = 1 
                break

    #print(quad)
    return quad


def quad(delivery, cell_ids):
    p = LatLng.from_degrees(delivery.point.lat, delivery.point.lng)
    for i in range(len(cell_ids)):
        if (LatLngRect.from_point(p).intersects(Cell(cell_ids[i]))):
            return i
    return -1


def data_frame(routes, cell_ids):
    data = []
    somas = []
    for i in range(len(routes.vehicles)):
        soma = somaQdR(routes.vehicles[i], cell_ids)
        data.append(soma)
        somas.append(soma)
    columns = [i for i in range(len(cell_ids))]
    return pd.DataFrame(data, columns=columns),somas



def has_same_quad(cell_ids, delivery_pnt, uc_pnt):
    p1 = LatLng.from_degrees(delivery_pnt.lat, delivery_pnt.lng)
    p2 = LatLng.from_degrees(uc_pnt.point.lat, uc_pnt.point.lng)
    for i in range(len(cell_ids)):
        if (LatLngRect.from_point(p1).intersects(Cell(cell_ids[i]))) and (LatLngRect.from_point(p2).intersects(Cell(cell_ids[i]))):
            return True
    return False

def has_same_quad2(cell_ids, delivery):
    p1 = LatLng.from_degrees(delivery.lat, delivery.lng)
    for i in range(len(cell_ids)):
        if (LatLngRect.from_point(p1).intersects(Cell(cell_ids[i]))):
            return True
    return False
                


def s2gen(max_lat, min_lat, max_lng, min_lng, s2_lvl):
    point_nw = LatLng.from_degrees(max_lat, min_lng)
    point_se = LatLng.from_degrees(min_lat, max_lng)

    rc = RegionCoverer()
    rc.min_level = s2_lvl
    rc.max_level = s2_lvl
    rc.max_cells = 1000000

    cellids = rc.get_covering(LatLngRect.from_point_pair(point_nw, point_se))

    return cellids


#percorrer todos pontos do pool aos pares com base em todas as entregas
def filter_route2(p, cell_ids):
    for i in range(len(p.vehicles)):
        for j in range(len(p.vehicles)):
            found = False
            if i == j or j >= len(p.vehicles) or i >= len(p.vehicles):
                continue
            #print(i, j, len(p.vehicles))
            for d1 in range(len(p.vehicles[i].deliveries)):
                for d2 in range(len(p.vehicles[j].deliveries)):
                    p1 = LatLng.from_degrees(p.vehicles[i].deliveries[d1].point.lat, p.vehicles[i].deliveries[d1].point.lng)
                    p2 = LatLng.from_degrees(p.vehicles[j].deliveries[d2].point.lat, p.vehicles[j].deliveries[d2].point.lng)
    
                    cell1 = 0
                    cell2 = 0
                    for k in range(len(cell_ids)):
                        if (LatLngRect.from_point(p1).intersects(Cell(cell_ids[k]))):
                            cell1 = i
                    for k in range(len(cell_ids)):
                        if (LatLngRect.from_point(p2).intersects(Cell(cell_ids[k]))):
                            cell2 = i 
                    #rotas similares 
                    if cell1 == cell2:
                        p.vehicles.remove(p.vehicles[j])
                        found = True
                        break
                if found:
                    break
    return p


#percorrer todos pontos do pool aos pares com base na primeira entrega
def filter_route(p, cell_ids):
    for i in range(len(p.vehicles)):
        for j in range(len(p.vehicles)):
            if i == j or j >= len(p.vehicles) or i >= len(p.vehicles):
                continue
            #print(i, j, len(p.vehicles))
            p1 = LatLng.from_degrees(p.vehicles[i].deliveries[0].point.lat, p.vehicles[i].deliveries[0].point.lng)
            p2 = LatLng.from_degrees(p.vehicles[j].deliveries[0].point.lat, p.vehicles[j].deliveries[0].point.lng)
            cell1 = 0
            cell2 = 0
            for k in range(len(cell_ids)):
                if (LatLngRect.from_point(p1).intersects(Cell(cell_ids[k]))):
                    cell1 = k
            for k in range(len(cell_ids)):
                if (LatLngRect.from_point(p2).intersects(Cell(cell_ids[k]))):
                    cell2 = k 
            #rotas similares 
            if cell1 == cell2:
                p.vehicles.remove(p.vehicles[j])
    return p


def max_min(p):
    max_lat = get_max_lat(p)
    min_lat = get_min_lat(p)
    max_lng = get_max_lng(p)
    min_lng = get_min_lng(p)
    return max_lat, min_lat, max_lng, min_lng




def eliminando_rotas_similares(p, nivel_s2cells):
    dfs = []
    for k in range(len(p)):
        #filtrando as rotas do pool
        max_lat, min_lat, max_lng, min_lng = max_min(p[k])
        cell_ids = s2gen(max_lat, min_lat, max_lng, min_lng, nivel_s2cells)
        df = data_frame(p[k], cell_ids)
        #p[k] = filter_route2(p[k], cell_ids)
        dfs.append(df)
    return dfs
