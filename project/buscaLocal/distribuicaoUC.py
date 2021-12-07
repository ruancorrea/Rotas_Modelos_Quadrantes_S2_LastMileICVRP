import math
from typing import Dict, List
from loggibud.v1.types import (
    CVRPSolution,
)

import numpy as np

# recolher informações como:
# a soma da distribuicao atual
# a chave do maior valor
# a chave do menor valor
def infos_distribute(arredondamento: Dict):
    soma = sum(filter(lambda elem:elem,(map(lambda dic:int(dic),arredondamento.values()))))
    chave_max = sorted(arredondamento, key=arredondamento.get, reverse=True)[0]
    chave_min = sorted(arredondamento, key=arredondamento.get)[0]
    return soma, chave_max, chave_min


def distributing(m: int, p: List[CVRPSolution], sum_p: int, ordenado: List[int]):
    ok = False
    minimo = False
    arredondamento = {i: int(np.ceil(m * len(p[i].deliveries)/sum_p)) for i in ordenado}
    #arredondamento = {i: round(m * len(p[i].deliveries)/sum_p) for i in ordenado}
    soma, chave_max, chave_min = infos_distribute(arredondamento)
    print("inicio",arredondamento)

    while ok == False or minimo == False:
    
        soma, chave_max, chave_min = infos_distribute(arredondamento)
        print("soma:", soma, "chave_max", chave_max, "chave_min", chave_min)
        
        if arredondamento[chave_min] == 0:
            arredondamento[chave_min] = arredondamento[chave_min] + 1
        else: 
            minimo = True
            if soma < m:
                arredondamento[chave_max] = arredondamento[chave_max] + 1
            
            elif soma > m:
                arredondamento[chave_max] = arredondamento[chave_max] - 1
            
            elif soma == m:
                ok = True
            
        xx = sorted(arredondamento, key=arredondamento.get, reverse=True)
        arredondamento = {i: arredondamento[i] for i in xx}
        print(arredondamento)

    print("distribuicao final",arredondamento)
    # transformando em list

    distribute = []
    for i in arredondamento:
        for j in range(arredondamento[i]):
            distribute.append(i)
    return distribute, arredondamento


# arredondamento pra cima com ceil
# assim, a verificação de algum cluster com 0 ucs não será mais necessária
# a soma será sempre maior ou igual ao número de ucs(m)
# com isso, para balancear, iremos precisar apenas remover ucs
# as ucs removidas serão daqueles clusters mais próximos de 1
# ex: c1: 3, c2: 2, c3: 1. caso precise remover, irá remover uma uc de c2.
def distributing2(m: int, p: List[CVRPSolution], sum_p: int, ordenado: List[int]):
    arredondamento = {i: int(np.ceil(m * len(p[i].deliveries)/sum_p)) for i in ordenado}
    soma, chave_max, chave_min = infos_distribute(arredondamento)
    xy = sorted(arredondamento, key=arredondamento.get)

    while soma != m:
        for i in xy:
            if arredondamento[i] > 1:
                arredondamento[i] = arredondamento[i] - 1
                break
        xx = sorted(arredondamento, key=arredondamento.get, reverse=True)
        arredondamento = {i: arredondamento[i] for i in xx}
        xy = sorted(arredondamento, key=arredondamento.get)
        soma, chave_max, chave_min = infos_distribute(arredondamento)
        print(arredondamento)

    print("distribuicao final",arredondamento)
    
    # transformando em list
    distribute = []
    for i in arredondamento:
        for j in range(arredondamento[i]):
            distribute.append(i)
    return distribute, arredondamento




def uc_distribute(m: int, p: List[CVRPSolution]):
    tam_pools = {i: len(p[i].deliveries) for i in range(len(p))}
    #tam_pools = {i: len(p[i].vehicles) for i in range(len(p))}
    ordenado = sorted(tam_pools, key = tam_pools.get, reverse=True)
    sum_clusters = sum([len(p[i].deliveries) for i in range(len(p))])
    distribuicao, dictDistribuicao = distributing2(m, p, sum_clusters, ordenado) # [1, 0, 2, 4]
    return distribuicao, dictDistribuicao