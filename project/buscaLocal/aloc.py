from platform import java_ver
import numpy as np
import math
import multiprocessing
import logging
import os

from copy import deepcopy
import time

from dataclasses import dataclass
from typing import Optional, List, Dict
from multiprocessing import Pool
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_array
from tqdm import tqdm

from loggibud.v1.types import (
    Delivery,
    CVRPInstance,
    CVRPSolution,
    CVRPSolutionVehicle,
    JSONDataclassMixin,
    Point,
)


from loggibud.v1.baselines.shared.ortools import (
    solve as ortools_solve,
    ORToolsParams,
)


from haversine import haversine
from loggibud.v1.eval.task1 import evaluate_solution

from project.buscaLocal.distances import dist, dist_route_delivery
from project.buscaLocal.distribuicaoUC import uc_distribute
from project.buscaLocal.s2cells import max_min, s2gen, has_same_quad, has_same_quad2
import matplotlib.pyplot as plt
from project.utils import createUC, AlocModel, AlocParams, create_instanceCVRP
#import pandas as pd
#from mlxtend.frequent_patterns import apriori
#from mlxtend.frequent_patterns import association_rules

from project.utils import create_instanceCVRP, dictOffilinePA0, dictOffilineDF0, dictOffilineRJ0

logger = logging.getLogger(__name__)



def dict_distances_centers(point, centers):
    dict_distances = {i: haversine((point.lat, point.lng), (centers[i][1], centers[i][0])) for i in range(len(centers))}
    return sorted(dict_distances, key = dict_distances.get)


def predict_centers(point, centers):
    d_min = math.inf
    i_min = 0

    for i in range(len(centers)):
        d = haversine((point.lat, point.lng), (centers[i][1], centers[i][0]))
        if d < d_min:
            d_min = d
            i_min = i
    
    return i_min

def pretrain(
    instances: List[CVRPInstance], pools: List[CVRPSolution], num_clusters: int, params: Optional[AlocParams] = None
) -> AlocModel:

    params = params or AlocParams.get_baseline()

    points = np.array(
        [
            [d.point.lng, d.point.lat]
            for instance in instances
            for d in instance.deliveries
        ]
    )

    points2 = np.array(
        [
            [d.point.lng, d.point.lat]
            for pool in pools
            for d in pool.deliveries
        ]
    )

    logger.info(f"Clustering instance into {num_clusters} subinstances")
    clustering = KMeans(num_clusters, random_state=params.seed)
    clustering.fit(points)

    nameClustering = f"clustering_{num_clusters}_{params.pts_batch}.png"
   # nameCenters = f"centers_{num_clusters}_{params.pts_batch}.png"
    nameClustering2 = f"clusteringFilter_{num_clusters}_{params.pts_batch}.png"

    clustering2 = KMeans(num_clusters, random_state=params.seed)
    clustering2.fit(points2)

    #plt.scatter(points[:, 0], points[:, 1], c = clustering.labels_, s=10)
    #plt.savefig (nameClustering)
   # plt.close()
    #plt.scatter(points2[:, 0], points2[:, 1], c = clustering2.labels_, s=10)
    #plt.savefig (nameClustering2)

    #centers = clustering.cluster_centers_
    #plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    #plt.savefig (nameCenters)

    return AlocModel(
                        params=params,
                        clustering = clustering
                    )



def quadranteCells(deliveries, sub, local, cell_ids, d_min, centroides):
    j_min = -1
    found = False
    for j in sub:
        if len(deliveries) > 0:
            for delivery_uc in deliveries[j]:
                if (has_same_quad(cell_ids, local, delivery_uc)):
                    d = dist(local, delivery_uc)
                    if d < d_min:
                        j_min = j
                        d_min = d
                        found = True
    
    return j_min, d_min, found


def quadranteCells2(deliveries, local, cell_ids, d_min):
    j_min = -1
    found = False
    for delivery_uc in deliveries:
        if (has_same_quad(cell_ids, local, delivery_uc)):
            d = dist(local, delivery_uc)
            if d < d_min:
                d_min = d
                found = True
    
    return d_min, found



def distanciaRM(deliveries, cluster, p, delivery): 
    rota_modelo = []
    d_min = math.inf
    qtd_max = 0
    e_copy = deepcopy(deliveries)
    e_copy.append(delivery)
    for k in range(len(p[cluster].vehicles)):
        qtd = 0
        if delivery in p[cluster].vehicles[k].deliveries:
            qtd += 1
            for deli in deliveries:
                if deli in p[cluster].vehicles[k].deliveries:
                    qtd += 1
            if qtd > qtd_max:
                qtd_max = qtd
                d = dist_route_delivery(e_copy, p[cluster].vehicles[k].deliveries)
                if d < d_min:
                    d_min = d
                    rota_modelo = p[cluster].vehicles[k].deliveries
    e_copyRM = []

    if len(rota_modelo) > 0:
        delete_deliveries = []
        e_copyRM = deepcopy(rota_modelo)
        for i in range(len(e_copyRM)):
            for deli in e_copy:
                if deli == e_copyRM[i]:
                    if i not in delete_deliveries:
                        delete_deliveries.append(i)
        delete_deliveries.sort(reverse=True)
       # print(delete_deliveries)
        for d in delete_deliveries:
            e_copyRM.remove(e_copyRM[d])

    return e_copyRM

def distanciaRM2(deliveries, cluster, p, delivery): 
    rota_modelo = -1
    d_min = math.inf
    rotas_analise = []

    for k in range(len(p[cluster].vehicles)):
        if delivery in p[cluster].vehicles[k].deliveries:
            copy = deepcopy(p[cluster].vehicles[k])
            for d in copy.deliveries:
                if d == delivery: 
                    copy.deliveries.remove(delivery)
            rotas_analise.append(copy)

    print(f"Tamanho rotas_modelos: {len(p[cluster].vehicles)} // rotas em analise: {len(rotas_analise)}")
    if len(rotas_analise) > 0:
        for rota in rotas_analise:
            e_copy = deepcopy(deliveries)
            e_copy.append(delivery)
            d = dist_route_delivery(e_copy, rota.deliveries)
            if d < d_min: 
                d_min = d
                rota_modelo = deepcopy(rota.deliveries)
   # if len(p[cluster].vehicles) > 20:
   #     rotasAleatorias = np.random.randint(len(p[cluster].vehicles),size=20)
   # else:
   #     rotasAleatorias = [i for i in range(len(p[cluster].vehicles))]
    else:
        for k in range(len(p[cluster].vehicles)):
    # for k in rotasAleatorias:
            # calculo distancia rota modelo
            e_copy = deepcopy(deliveries)
            e_copy.append(delivery)
            d = dist_route_delivery(e_copy, p[cluster].vehicles[k].deliveries)
            if d < d_min: 
                d_min = d
                rota_modelo = deepcopy(p[cluster].vehicles[k].deliveries)


    return rota_modelo

def calculoRota(delivery, rota_modelo):
    d_min = math.inf
    for deli in rota_modelo:
        d = haversine((delivery.point.lat, delivery.point.lng), (deli.point.lat, deli.point.lng))
        if d>0 and d<d_min:
            d_min = d

    return d_min

def distanciaRM3(deliveries, cluster, p, delivery): 
    copy = deepcopy(deliveries)
    copy.append(delivery)
    qtd_max = 0
    rota_modelo = []
    for k in range(len(p[cluster].vehicles)):
        qtd = 0
        for d in copy:
            if d in p[cluster].vehicles[k].deliveries:
                qtd = qtd + 1 
        if qtd > qtd_max:
            qtd_max = qtd
            rota_modelo = deepcopy(p[cluster].vehicles[k])
    delete_deliveries = []
    for i in range(len(rota_modelo.deliveries)):
        for d in copy:
            if d == rota_modelo.deliveries[i] : 
                delete_deliveries.append(i)
    delete_deliveries.sort(reverse=True)
    for i in delete_deliveries:
        rota_modelo.deliveries.remove(rota_modelo.deliveries[i])
    print(f"QUANTIDADE MAXIMA ENCONTRADA: {qtd_max}")
    return rota_modelo.deliveries

def distanciaUC(deliveries, phi, sub, local, d_min, j_min, k_min):
    for j in sub: 
        if (len(deliveries[j]) > 0): 
            # calculo entre os pontos das ucs
            d = min(dist(local, delivery) for delivery in deliveries[j]) 
            if d < d_min: 
                j_min = j 
                d_min = d
                k_min = phi[j]

    return j_min, d_min, k_min



def entregasIniciais(distribuicao, model, pools_instances, dictDistribuicao):
    entregas = [[] for i in range(len(distribuicao))]

    for i in range(len(pools_instances)):
        sub = [j for j in range(len(distribuicao)) if distribuicao[j] == i]
        points = np.array(
            [
                [d.point.lng, d.point.lat]
                for d in pools_instances[i].deliveries
            ]
        )
        num_clusters = dictDistribuicao[i]

        model = KMeans(num_clusters, random_state=0).fit(points)
        for j in range(num_clusters):
            #entregas[sub[j]] = model.cluster_centers_[j]
            point = Point(model.cluster_centers_[j][0], model.cluster_centers_[j][1])
            entregas[sub[j]] = Delivery(id=f"temp{sub[j]}", point=point, size=1)
    return entregas



def centroidsProximos(centers, sub, delivery):
    qtd = 3 if len(sub) > 3 else len(sub)
    dict_distances = {i: haversine((delivery.point.lat, delivery.point.lng), (centers[i].point.lat, centers[i].point.lng)) for i in sub}
    ordenado = sorted(dict_distances, key = dict_distances.get)
    return ordenado[:qtd]


# ruan 4/8/21
def aloc(instance: CVRPInstance, p: List[CVRPSolution], 
model, params: AlocParams, distribuicao: List[int], cell_ids, centroides, dictDistribuicao) -> CVRPSolution:
    #UCS = [createUC() for i in range(params.num_ucs)]
    C = [0 for i in range(params.num_ucs)] # capacidade da ucj
    phi = [[] for i in range(params.num_ucs)] # rota modelo de ucj
    deliveries = [[] for i in range(params.num_ucs)] # entregas associadas a ucj
    R = [] # conjunto de entregas   
    vehicles = []
    contS2 = 0
    contRM = 0
    contPHI = 0
    c1 = 0

    for delivery in instance.deliveries:
        # predict
        cluster = model.clustering.predict([[delivery.point.lng, delivery.point.lat]])[0]
        sub = [i for i in range(len(distribuicao)) if distribuicao[i] == cluster]
        rota_modelo = []
        key = dictDistribuicao[cluster]
        if key == 1:
            j_min = sub[0]
            c1 += 1
        else:
            centrsProximos = centroidsProximos(centroides, sub, delivery)
            d_min = math.inf
            found1 = False
            for j in sub:
                if len(phi[j]) > 0:
                    if delivery in phi[j]:
                        found1 = True
                        j_min = j
                        rota_modelo = phi[j]
                        break
            if found1:
                contPHI += 1
            
            if not found1:
                j_min, d_min, found2 = quadranteCells(deliveries, centrsProximos, delivery.point, cell_ids[cluster], d_min, centroides)               
            #found2 = False

            if found2:
                #logger.info(f"Entrega no mesmo quadrante de outra já alocada.")
                contS2 += 1 
                rota_modelo = []

            if not found2 and not found1:
                d_min = math.inf
                for j in centrsProximos:
                    d = dist(delivery.point, centroides[j])
                    rota_modelo = []
                    if len(deliveries[j]) > 0:
                        rota_modelo = distanciaRM(deliveries[j], cluster, p, delivery)
                        contRM += 1 
                        ecopy = deepcopy(rota_modelo)
                        for deli in deliveries[j]:
                            ecopy.append(deli)
                        d1 = calculoRota(delivery, ecopy)
                        if d1 < d:
                            d = d1

                    if d < d_min: 
                        j_min = j 
                        d_min = d 

        #j_min, d_min, k_min = distanciaUC(deliveries, phi, sub, delivery.point, d_min, j_min, rota_modelo)
        
        #print(f"cluster: {cluster}", end= " ")
        #for j in sub:
        #    print(f"uc {j}: {len(deliveries[j])}  || ", end= " ")
        #print("\n")

        # uc sendo despachada
        if (C[j_min] + delivery.size) > instance.vehicle_capacity:
            logger.info(f"Despachando Unidade de Carregamento {j_min}.")
            #print(f"Despachando UC: C {C[j_min]} / d {delivery.size} ")
            inst = create_instanceCVRP(instance, deliveries[j_min], instance.name, 3)
            R.append(inst)
            C[j_min] = 0
            deliveries[j_min] = []

        C[j_min] = C[j_min] + delivery.size
        deliveries[j_min].append(delivery)
        phi[j_min] = deepcopy(rota_modelo) if len(rota_modelo) > 0 else phi[j_min]
        #if rota_modelo not in phi[j_min] and len(rota_modelo) > 0 :
        #    phi[j_min].append(deepcopy(rota_modelo))

    logger.info("Despachando Unidades de Carregamento que não chegaram ao limite.")
    # ucs que nao foram despachadas
    for j in range(params.num_ucs):
        if C[j] > 0:
            inst = create_instanceCVRP(instance, deliveries[j], instance.name, 3)
            R.append(inst)
            C[j] = 0
            deliveries[j] = []

    logger.info("Organizando rotas")
    # organizando rotas            
    for inst in R:
        sol = ortools_solve(inst, params.ortools_tsp_params)# TSP
        while not isinstance(sol, CVRPSolution):
            logger.info(f"SOLUÇÃO NONETYPE. Buscando novamente. {instance.name}")
            sol = ortools_solve(inst, params.ortools_tsp_params)# TSP
        vehicles.append(CVRPSolutionVehicle(instance.origin, sol.deliveries))

    logger.info(f"S2 CELLS: {contS2}  |  ROTA MODELO: {contRM}.   |  PHI: {contPHI} | 1 UC: {c1}")

    return CVRPSolution(
        name=instance.name,
        vehicles= vehicles,
    )

def poolClusters(model, pools):
    for i in range(len(pools)):
        print(pools[i].name)
    clusters =  {model.clustering.predict([[pools[i].deliveries[0].point.lng, pools[i].deliveries[0].point.lat]])[0]: i for i in range(len(pools))}
    print(clusters)
    return clusters

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser()

    parser.add_argument("--eval_instances", type=str, required=True)
    parser.add_argument("--train_instances", type=str, required=True)    
    parser.add_argument("--pools", type=str, required=True)
    parser.add_argument("--output", type=str)
    parser.add_argument("--params", type=str)

    args = parser.parse_args()

    # Load instance and heuristic params.
    eval_path = Path(args.eval_instances)
    eval_path_dir = eval_path if eval_path.is_dir() else eval_path.parent
    eval_files = (
        [eval_path] if eval_path.is_file() else list(eval_path.iterdir())
    )

    train_path = Path(args.train_instances)
    train_path_dir = train_path if train_path.is_dir() else train_path.parent
    train_files = (
        [train_path] if train_path.is_file() else list(train_path.iterdir())
    )
    train_instances = [CVRPInstance.from_file(f) for f in train_files[:240]]

    pools_path = Path(args.pools)
    pools_path_dir = pools_path if pools_path.is_dir() else pools_path.parent
    pools_files = (
        [pools_path] if pools_path.is_file() else list(pools_path.iterdir())
    )

    pools_instances = [[] for i in range(len(pools_files))]

   # pools_instances = [CVRPSolution.from_file(f) for f in pools_files[:240]]
    for f in pools_files[:240]:
        file = CVRPSolution.from_file(f)
        sp = file.name.split("_")
        cluster = int(sp[1])
        pools_instances[cluster] = file

    for i in range(len(pools_instances)):
        print(f"{i} : {pools_instances[i].name}")
     
    params = AlocParams.from_file(args.params) if args.params else AlocParams.get_baseline()
    print(params)

    output_dir = Path(args.output or ".")
    output_dir.mkdir(parents=True, exist_ok=True)

    num_points = params.pts_batch
    logger.info("Fazendo a distribuição das Unidades de Carregamento.")

    distribuicao, dictDistribuicao = uc_distribute(params.num_ucs, pools_instances)

    model = pretrain(train_instances, pools_instances, len(pools_instances), params)
    #clusters = poolClusters(model, pools_instances)

    #assert len(clusters) == len(pools_instances)
    logger.info(f"Buscando cedulas S2sphere.")
    cell_ids = []
    for i in range(model.clustering.n_clusters):
        max_lat, min_lat, max_lng, min_lng = max_min(pools_instances[i])
        ids = s2gen(max_lat, min_lat, max_lng, min_lng, params.nivel_s2cells)
        cell_ids.append(ids)

    #for id in cell_ids:
    #    print(id)
    #    print()

    solutions = []
    manager = multiprocessing.Manager()
    results = manager.list()
    print("tam pools", sum([len(pools_instances[i].vehicles) for i in range(len(pools_instances))]))
    if len(pools_instances) > params.num_ucs:
        print("ERROR! Número de unidades de carregamento menor que o número de clusters.")
    centroids = entregasIniciais(distribuicao, model, pools_instances, dictDistribuicao)

    def solve(file):
        instance = CVRPInstance.from_file(file)
        logger.info("Alocando entregas")
        solution_on = aloc(instance, pools_instances, model, params, distribuicao, cell_ids, centroids, dictDistribuicao) 
       # solution.to_file((output_dir / f"{instance.name}.json"))
       # print(f"solution:{len(solution_on.deliveries)} // instance:{len(instance.deliveries)}")

        distance_on = evaluate_solution(instance, solution_on)
        res = (instance.name, distance_on)
        results.append(res)

    inicio = time.time()
    # caso haja problema no tqdm
    for eval in eval_files:
        solve(eval)

    # Run solver on multiprocessing pool.
    #with Pool(os.cpu_count()) as pool:
    #    list(tqdm(pool.imap(solve, eval_files), total=len(eval_files)))

    final = time.time()
    tmp = final - inicio

    print(pools_path)
    porcs = []
    for instance, distance in results:
        porc = (distance/dictOffilinePA0[instance])*100 - 100
        porcs.append(porc)
        print(f"{instance}: {porc} %")
    soma = 0
    for p in porcs:
        soma += p
    
    print("media:",soma/len(porcs))
    print("tempo: ", tmp)
