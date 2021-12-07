from copy import deepcopy
from multiprocessing import Pool
from typing import List
from tqdm import tqdm

from pandas.core.frame import DataFrame
from project.buscaLocal.s2cells import data_frame, s2gen, max_min, quads_of_route
import logging
import os
from argparse import ArgumentParser
from pathlib import Path

from loggibud.v1.types import (
    CVRPSolution,
)

logger = logging.getLogger(__name__)

def Somatorio(df: DataFrame):
    somatorio = {r: 0 for index, row in df.iterrows() for r in range(len(row)) if row[r] > 0}
    for index, row in df.iterrows():
        for r in range(len(row)):
            if row[r] > 0: 
                somatorio[r] += row[r]
                #print(f"({r},{row[r]})")
    return somatorio


def quadrantesFrequentes(dfs: List[DataFrame]):
    quadrantes_frequentes = []
    for i in range(len(dfs)):
        somatorio = Somatorio(dfs[i])
        quadrantes_frequentes.append(somatorio)
        #print()
    return quadrantes_frequentes


def quadrantesRestantes(quadrantes_frequentes, pools: List[CVRPSolution], porc: int):
    somatorio = {q: quadrantes_frequentes[q]/len(pools.deliveries) for q in quadrantes_frequentes}
    xx = sorted(somatorio, key=somatorio.get, reverse=True)
    somatorio = {i: somatorio[i]*100 for i in xx}
    media = porc/len(xx)
    # media = 3
    acima_da_media = []
    for i in xx:
        if somatorio[i] > media:
            acima_da_media.append(i)
    return acima_da_media


def escolhendoVeiculos(pools: List[CVRPSolution], cell_ids, quadrantes_restantes: List[int], quads ):
    remove_vehicle = []
    for v in range(len(pools.vehicles)):
        #quads = quads_of_route(pools.vehicles[v], cell_ids)
        for q in range(len(quads[v])):
            if quads[v][q] > 0 and q not in quadrantes_restantes: 
                #print(v)
                remove_vehicle.append(v)
                break
    return remove_vehicle



def gerandoDataFramesCellsIDS(pools_instances: List[CVRPSolution], nivelS2: int):
    cell_ids = []
    dfs = []
    quads = []
    for k in range(len(pools_instances)):
    #filtrando as rotas do pool
        max_lat, min_lat, max_lng, min_lng = max_min(pools_instances[k])
        cell_id = s2gen(max_lat, min_lat, max_lng, min_lng, nivelS2)
        cell_ids.append(cell_id)
        df, somas = data_frame(pools_instances[k], cell_id)
        quads.append(somas)
        dfs.append(df)
    
    return dfs, cell_ids, quads


def RemovendoRotas(pool: CVRPSolution, quadrantes_frequentes: List[int], fats: List[int], j: int, cell_ids, quads):
    print(f"DE {len(pool.vehicles)}")
    igualdade = True
    remove_vehicle = []

    while igualdade and j < len(fats):
        acima_da_media = quadrantesRestantes(quadrantes_frequentes, pool, fats[j])
        remove_vehicle = escolhendoVeiculos(pool, cell_ids, acima_da_media, quads)
        print(f"remove_vehicle_id: tam {len(remove_vehicle)}")
        j += 1
        #print(f"{j}: {remove_vehicle}")
        if len(pool.vehicles) != len(remove_vehicle):
            igualdade = False
    
    if len(pool.vehicles) == len(remove_vehicle) and j == len(fats):
        print("Tentativa de diminuir falhou")
        remove_vehicle = []

    return remove_vehicle


def filtragem(pools_instances: List[CVRPSolution], nivelS2: int) -> List[CVRPSolution]:
    logger.info(f"Iniciando Filtrando com nível {nivelS2} do s2sphere.")   
    logger.info("Quantidade rotas de cada pool.")   
    for p in range(len(pools_instances)):
        print(f"{p}: {len(pools_instances[p].vehicles)}    ", end= " ")
    print()

    # gera os data frames e armazena cell_ids
    logger.info("Organizando data frames e celulas s2.")   
    dfs, cell_ids, quads = gerandoDataFramesCellsIDS(pools_instances, nivelS2)

    # identifica os quadrantes frequentes
    # somamos a quantidade de entregas que estão presentes naquele determinado quadrante
    logger.info("Cálculando os quadrantes frequentes.")   
    quadrantes_frequentes = quadrantesFrequentes(dfs)
    
    # mesma ideia da distribuicao da UC
    # os quadrantes que ficarem abaixo da media serão considerados de baixa frequencia
    # acima da média serão considerados alta frequencia
    for i in range(len(pools_instances)):
        logger.info(f"Removendo rotas do pool {i}.")   
        fats = [150, 100, 75, 50, 25]
        j = 1 if len(pools_instances[i].vehicles) < 100 else 0
        #j = 1
        remove_vehicle = RemovendoRotas(pools_instances[i], quadrantes_frequentes[i], fats, j,  cell_ids[i], quads[i])
        remove_vehicle.sort(reverse=True)
        for v in remove_vehicle:
            pools_instances[i].vehicles.remove(pools_instances[i].vehicles[v])
        #print(remove_vehicle)
        print(f"AGORA {len(pools_instances[i].vehicles)}")

        while len(pools_instances[i].vehicles) > 20:
            x = len(pools_instances[i].vehicles) - 1
            pools_instances[i].vehicles.remove(pools_instances[i].vehicles[x])

        print(f"FINALIZADO {len(pools_instances[i].vehicles)}")
        print()


    return pools_instances




def filterRM(p: CVRPSolution) -> CVRPSolution:
    logger.info(f"Removendo rotas de {p.name}.")  
    max_lat, min_lat, max_lng, min_lng = max_min(p)

    logger.info(f"Gerando as celulas S2 de {p.name}.")  
    cell_ids = s2gen(max_lat, min_lat, max_lng, min_lng, nivelS2)

    logger.info(f"Gerando o data frame de {p.name}.")  
    df, quads = data_frame(p, cell_ids)

    logger.info(f"Cálculando os quadrantes frequentes de {p.name}.") 
    quadrantes_frequentes = Somatorio(df)  

    logger.info(f"Removendo rotas de {p.name}.")  
    fats = [150, 100, 75, 50, 25]
    j = 1 if len(p.vehicles) < 100 else 0
    remove_vehicle = RemovendoRotas(p, quadrantes_frequentes, fats, j,  cell_ids, quads)
    remove_vehicle.sort(reverse=True)
    for v in remove_vehicle:
        p.vehicles.remove(p.vehicles[v])
    #print(remove_vehicle)
    print(f"AGORA {len(p.vehicles)}")
    
    #while len(p.vehicles) > 20:
    #    x = len(p.vehicles) - 1
    #    p.vehicles.remove(p.vehicles[x])

    print(f"FINALIZADO {len(p.vehicles)}")
    print()

    return p


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser()

    parser.add_argument("--pools", type=str, required=True)
    parser.add_argument("--output", type=str)
    parser.add_argument("--params", type=str)

    args = parser.parse_args()
    
    logger.info("Carregando rotas modelos.")

    pools_path = Path(args.pools)
    pools_path_dir = pools_path if pools_path.is_dir() else pools_path.parent
    pools_files = (
        [pools_path] if pools_path.is_file() else list(pools_path.iterdir())
    )

    pools_instances = [CVRPSolution.from_file(f) for f in pools_files[:240]]


    nivelS2 = 12
    sp = str(pools_path).split("_")
    out = f"{sp[0]}Filter{nivelS2}_{sp[1]}_{sp[2]}"

    output = out if args.output else None
    output_dir = Path(output or ".")
    output_dir.mkdir(parents=True, exist_ok=True) 

   # pools_instances = filtragem(pools_instances, nivelS2)

    #for i in range(len(pools_instances)):
    #    pools_instances[i].to_file(output_dir / f"{pools_instances[i].name}.json")

    logger.info(f"Iniciando filtragem com nível {nivelS2} do s2sphere.")   
    def solve(file):
        pool = CVRPSolution.from_file(file)
        logger.info(f"Quantidade rotas de {pool.name}: {len(pool.vehicles)}")
        pool = filterRM(pool)   
        pool.to_file(output_dir / f"{pool.name}.json")

    with Pool(os.cpu_count()) as pool:
        list(tqdm(pool.imap(solve, pools_files), total=len(pools_files)))
    
    logger.info("Filtragem finalizada. Rotas modelo armazenadas.")   
