import logging
import os

from dataclasses import dataclass
from typing import Optional, List, Dict
from multiprocessing import Pool, Manager
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from copy import deepcopy

from loggibud.v1.types import (
    Delivery,
    CVRPInstance,
    CVRPSolution,
    CVRPSolutionVehicle,
    Point,
    JSONDataclassMixin,
)
from loggibud.v1.baselines.shared.ortools import (
    solve as ortools_solve,
    ORToolsParams,
)


import matplotlib.pyplot as plt
from project.utils import batchInstance, create_instanceCVRP
from math import sqrt


logger = logging.getLogger(__name__)

@dataclass
class CluteringParams(JSONDataclassMixin):
    num_clusters: Optional[int] = None
    num_entregas_batch: Optional[int] = 1000
    seed: int = 0
    @classmethod
    def get_baseline(cls):
        return cls(
            seed = 0,
            num_entregas_batch = 1000,

        )

@dataclass
class ClusteringModel:
    params: CluteringParams
    clustering: KMeans
    subinstance: Optional[CVRPInstance] = None
    cluster_subsolutions: Optional[Dict[int, List[CVRPSolutionVehicle]]] = None



def pretrain(
    instances: List[CVRPInstance], params: Optional[CluteringParams] = None
) -> ClusteringModel:
    params = params or CluteringParams.get_baseline()

    points = np.array(
        [
            [d.point.lng, d.point.lat]
            for instance in instances
            for d in instance.deliveries
        ]
    )

    num_clusters = params.num_clusters if params.num_clusters else metodoCotovelo(points)

    logger.info(f"Clustering instance into {num_clusters} subinstances")
    clustering = KMeans(num_clusters, random_state=params.seed)
    clustering.fit(points)

    return ClusteringModel(
        params=params,
        clustering=clustering,
    )


def numero_cluster(error_rate):
    x1, y1 = 1, error_rate[0]
    x2, y2 = 28, error_rate[len(error_rate)-1]
    logger.info("Calculando as distancias referente ao error_rate")
    distances = []
    for i in range(len(error_rate)):
        x0 = i+2
        y0 = error_rate[i]
        numerator = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    logger.info("Retornando o número de cluster.")
    return distances.index(max(distances)) + 2


def metodoCotovelo(points):
    logger.info("Calculando o error_rate")
    error_rate = []
    for i in range(2,29):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(points)
        error_rate.append(kmeans.inertia_)
    #name = "error_rate_df-0"
    #logger.info("Plotando o error_rate")

    #plt.plot(range(2, 29), error_rate, color="blue")
    #plt.savefig (name)
    return numero_cluster(error_rate)


def entregasDuplicadas(deliveries):
    points = [[delivery.point.lng, delivery.point.lat] for delivery in deliveries]
    pointsFilter = []
    deliveriesFilter = []
    for i in range(len(points)):
        if points[i] not in pointsFilter:
            pointsFilter.append(points[i])
            if points[i][0] == deliveries[i].point.lng:
                deliveriesFilter.append(deliveries[i])

    logger.info("Remoção das entregas duplicadas")
    logger.info(f"pointsFilter: {len(pointsFilter)} deliveriesFilter: {len(deliveriesFilter)}")

    return deliveriesFilter



if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser()

    parser.add_argument("--train_instances", type=str, required=True)
    parser.add_argument("--output", type=str)
    parser.add_argument("--params", type=str)

    args = parser.parse_args()

    train_path = Path(args.train_instances)
    train_path_dir = train_path if train_path.is_dir() else train_path.parent
    train_files = (
        [train_path] if train_path.is_file() else list(train_path.iterdir())
    )


    params = CluteringParams.from_file(args.params) if args.params else CluteringParams.get_baseline()
    print(params)
    train_instances = [CVRPInstance.from_file(f) for f in train_files[:240]]

    logger.info("Pretraining on training instances.")
    model = pretrain(train_instances, params)

    out = f"{args.output}/clusters1_{model.clustering.n_clusters}_{params.num_entregas_batch}" if args.output else None
    outInstances = f"{args.output}/instances" if args.output else None
    output_dir = Path(out or ".")
    output_dir.mkdir(parents=True, exist_ok=True)

    # juntando todas as entregas de todas as instancias
    deliveries = np.array(
        [
            d
            for instance in train_instances
            for d in instance.deliveries
        ]
    )

    logger.info(f"Total de entregas inicial: {len(deliveries)}")

    #deliveries = entregasDuplicadas(deliveries)

    #logger.info(f"Total de entregas pós remoção: {len(deliveries)}")

    logger.info("Separando as entregas em seus respectivos clusters")
    points_clusters = [[] for i in range(model.clustering.n_clusters)]
    for delivery in deliveries:
        cluster = model.clustering.predict([[delivery.point.lng, delivery.point.lat]])[0]
        points_clusters[cluster].append(delivery)

    logger.info("Criando as instances_clusters")
    instances = []
    for cluster in range(len(points_clusters)):
        name = f"cluster_{cluster}"
        instanceCluster = create_instanceCVRP(train_instances[0], points_clusters[cluster], name, 1)
        instances.append(instanceCluster)
        print(len(instanceCluster.deliveries))
    print(len(instances))

    logger.info("Separando os batchs e criando as instancias_batchs de cada instancia_clusters")
    # separando os batchs e criando as instancias_batchs de cada instancia_clusters
    instances_batchs = []
    for instance in instances:
        num_batchs = int(np.ceil(len(instance.deliveries)/params.num_entregas_batch)) # 250 1240     
        n = 0
        for j in range(num_batchs):
            n, batch_points = batchInstance(instance, n, params.num_entregas_batch)
            name_instance = f"batch_{j}_{instance.name}"
            instance_batch = create_instanceCVRP(instance, batch_points, name_instance, 1)
            instances_batchs.append(instance_batch)

    print(len(instances_batchs))

    manager = Manager()
    pools_solutions = [manager.list() for i in range(model.clustering.n_clusters)]

    # resolvendo cada instancia_batch de cada instancia_cluster chamando o CVRP do ortools
    def solve(instance: CVRPInstance):
#        instance.to_file(outInstances / f"{instance.name}.json")
        logger.info(f"Resolvendo {instance.name} chamando o CVRP")
        if len(instance.deliveries) > 0:
            solution = ortools_solve(instance)
            while not isinstance(solution,CVRPSolution):
                solution = ortools_solve(instance)
            if isinstance(solution, CVRPSolution):
                ids = instance.name.split("_")
                cluster = int(ids[len(ids)-1])
                if (len(solution.vehicles) > 0):
                    for vehicle in solution.vehicles:
                        pools_solutions[cluster].append(CVRPSolutionVehicle(instance.origin,vehicle.deliveries))


    # Run solver on multiprocessing pool.
    with Pool(os.cpu_count()) as pool:
        list(tqdm(pool.imap(solve, instances_batchs), total=len(instances_batchs)))

    logger.info("Verificando quantidade de entregas solucionadas")
    soma_clusters = sum(len(vehicles.deliveries) for pool in pools_solutions for vehicles in pool)
    print(f"soma_clusters {soma_clusters} // len(deliveries) {len(deliveries)} ")
    assert soma_clusters == len(deliveries)

    pools_instances = []
    logger.info("Juntando as soluções em cada arquivo_cluster")
    for cluster in range(len(pools_solutions)):
        name = f"cluster_{cluster}"
        vehicles = deepcopy(pools_solutions[cluster])
        solution = CVRPSolution(name=name, vehicles=vehicles)
        pools_instances.append(solution)
        #solution.to_file(output_dir / f"{name}.json")
    


    #pools_instances = filtragem(pools_instances)   

    for i in range(len(pools_instances)):
        pools_instances[i].to_file(output_dir / f"{pools_instances[i].name}.json")