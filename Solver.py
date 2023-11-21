from pathlib import Path
import json
import numpy as np
from Common import *
from queue import PriorityQueue
import itertools
import time
import torch
import cvxpy as cp
from gekko import GEKKO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_ITER_GEKKO = 2000 # max number of iteration in gekko

class Solver:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.W = self.settings['ffrMask']
    @classmethod
    def GetDefaultSettings(cls) -> dict:
        return {
            # 'BB': branch and bound
            # 'BB' degenerates to 1-step greedy when maxNumNodes = M
            # '0G': 0-step simple greedy, which observes once and greedy selection
            # 'Uni': select candidates uniformly in order
            # 'UnB': uniform then branch and bound, searching around the terminal states only
            
            # 'C2G': C2G alg in IXR (kmax=2)
            # 'C2I': C2I alg in IXR (kmax=3)
            'policy': 'BB',
            'a': None, # a in 1 - e^(-ax) (pixel activation)
            'ffrMask': 1.0, # mask for weighting the pixels
        }
    def summary(self, outDir: Path):
        outDir.mkdir(exist_ok=True)
        with open(outDir/'Solver.json', 'w') as f:
            j = dict(self.settings)
            if isinstance(j['ffrMask'], float) == False:
                j['ffrMask'] = j['ffrMask'].tolist()
            json.dump(j, f)
    def opt(self, S, C):
        '''
        * S : (M,) selection, must be np.array
        * C : same as C in self.solve
        force_np == True requires that C to be np.array
        '''
        a = self.settings['a']
        if device == 'cpu':
            # cvgCounts = S[:, None, None, None] * C # (M, M, h, w)
            cvg = np.sum(S[:, None, None, None] * C, axis=0) # (M, h, w)
            return np.sum(self.W * pixelActivation(cvg, a)) / S.shape[0]
        else:
            S = torch.from_numpy(S).to(device)
            cvg = torch.sum(S[:, None, None, None] * C, axis=0) # (M, h, w)
            return torch.sum(self.W * pixelActivation(cvg, a)).detach().item() / S.shape[0]
    def solve(self, C, N, maxNumNodes: int, callbacks: dict):
        '''
        C: (M, M, H, W) coverage map, either lower bound, exact or upper bound
        N: number of 1s allowed
        solQual: break solution if Lb >= solQual * Ub
        return sol (M,) , opt, ub
        '''
        M = C.shape[0]
        policy = self.settings['policy']
        if device != 'cpu':
            C = torch.from_numpy(C).to(device)
        if isinstance(self.W, float):
            self.W = np.ones(C.shape[2:], dtype=np.float32)
            self.W /= self.W.size
            if device != 'cpu':
                self.W = torch.from_numpy(self.W).to(device)
        elif isinstance(self.W, np.ndarray):
            if device != 'cpu':
                self.W = torch.from_numpy(self.W).to(device)
        if policy == 'BB':
            lastCount, count = 1, 0
            pq = PriorityQueue()
            nodeCount = itertools.count(0)
            maxLb = 0
            solNode = (
                -maxLb, # lb
                -self.opt(np.ones((M,), dtype=bool), C), # ub
                next(nodeCount),
                np.zeros((M,), dtype=bool), # sel
                [i for i in range(M)], # TBD
            )
            pq.put(solNode)
            while (not pq.empty()) and (count <= maxNumNodes):
                node = pq.get()
                ubSel = np.array(node[3], dtype=bool)
                ubSel[node[4]] = 1
                lb = -node[0]
                ub = -node[1]
                if "onOptUpdate" in callbacks:
                    callbacks["onOptUpdate"](lb)
                maxLb = max(maxLb, lb) # for maxLb to record the first node
                count += 1
                if count >= 2 * lastCount:
                    print(f'{count} nodes expanded')
                    lastCount = count
                if count > maxNumNodes:
                    break
                if ub < maxLb or np.sum(node[3]) >= N: # bound
                    continue
                else: # branch
                    for expandIdx in node[4]:
                        sol = np.array(node[3], dtype=bool)
                        sol[expandIdx] = 1 # select this
                        ubSel = np.array(sol, dtype=bool)
                        ubSel[node[4]] = 1
                        nextToBedetermined = list(node[4])
                        nextToBedetermined.remove(expandIdx)
                        nodeLb = self.opt(sol, C)
                        nodeUb = self.opt(ubSel, C)
                        nodeNext = (
                            -nodeLb,
                            -nodeUb,
                            next(nodeCount),
                            sol, 
                            nextToBedetermined
                        )
                        pq.put(nodeNext)
                        if nodeLb >= maxLb:
                            solNode = nodeNext
                            maxLb = nodeLb
            # sol, lb, ub
            return solNode[3], -solNode[0], -solNode[1]
        elif policy == '0G':
            contri = np.zeros((M,))
            for i in range(M):
                s = np.zeros((M,), dtype=bool)
                s[i] = 1
                contri[i] = self.opt(s, C)
            selectIdx = np.argsort(contri)[-N:]
            s = np.zeros((M,), dtype=bool)
            s[selectIdx] = 1
            
            ss = np.zeros((M,), dtype=bool)
            indices = selectIdx[::-1]
            for idx in indices[:N]:
                ss[idx] = 1
                if "onOptUpdate" in callbacks:
                        callbacks["onOptUpdate"](self.opt(ss, C))
            opt = self.opt(s, C)
            # sol, lb, ub
            return s, opt, opt
        elif policy == 'Uni':
            s = np.zeros((M,), dtype=bool)
            idx = round(M/N/2)
            count = 0
            while idx < M and count < N:
                s[min(round(idx), M-1)] = 1
                idx += M/N
                count += 1
                if "onOptUpdate" in callbacks:
                    callbacks["onOptUpdate"](self.opt(s, C))
            assert np.sum(s) == N
            opt = self.opt(s, C)
            return s, opt, opt
        elif policy == 'UnB':
            '''
            try to efficiently update the sol in B^N_M space
            clear the bit which keeps maximum opt
            set the bit which keeps maximum opt after clearing one bit
            '''
            # from policy == 'Uni'
            s = np.zeros((M,), dtype=bool)
            idx = round(M/N/2)
            count = 0
            while idx < M and count < N:
                s[min(round(idx), M-1)] = 1
                idx += M/N
                count += 1
            assert np.sum(s) == N
            # 'UnB'
            count = 0
            opt = self.opt(s, C)
            sol = s.copy()
            bestSol = sol.copy()
            bestOpt = opt
            searched = set()
            searched.add(tuple(sol.tolist()))
            print(f'UnB opt = {opt}')
            # since we are evaluating two linear times
            # we divide the nodes allowed by 2 to make it comparable to Naive BB
            while count <= maxNumNodes * M / (N + M):
                if "onOptUpdate" in callbacks:
                    callbacks["onOptUpdate"](bestOpt)
                toClrIndices = np.nonzero(sol)[0]
                clrRes = []
                toSetIndices = np.nonzero(np.logical_not(sol))[0]
                setRes = []
                
                for idx in toClrIndices:
                    s = sol.copy()
                    s[idx] = 0
                    clrRes.append([idx, self.opt(s, C)])
                clrRes = np.array(clrRes) # Nx2
                
                clrArgSort = np.argsort(clrRes[:, 1])[::-1] # descending
                clrRes = clrRes[clrArgSort]
                # dry-run to ensure that the clear-set sequence in the following computation
                # has at least one feasible update
                clrIdx = None
                breakFlag = False
                for idx in clrRes[:, 0]:
                    idx = int(idx)
                    if breakFlag:
                        break
                    for sidx in toSetIndices:
                        sidx = int(sidx)
                        s = sol.copy()
                        s[idx] = 0
                        s[sidx] = 1
                        if (tuple(s.tolist()) in searched) == False:
                            clrIdx = idx
                            breakFlag = True
                            break
                if breakFlag == False:
                    print('fully searched')
                    return bestSol, bestOpt, bestOpt
                
                ss = sol.copy()
                ss[clrIdx] = 0
                
                for idx in toSetIndices:
                    s = ss.copy()
                    s[idx] = 1
                    setRes.append([idx, self.opt(s, C)])
                setRes = np.array(setRes) # Nx2
                setArgSort = np.argsort(setRes[:, 1])[::-1] # descending
                setRes = setRes[setArgSort, :] # (setIdx, opts)
                
                # set next state
                setIdx = int(setRes[0, 0])
                for idx in setRes[:, 0]:
                    idx = int(idx)
                    s = ss.copy()
                    s[idx] = 1
                    if (tuple(s.tolist()) in searched) == False:
                        setIdx = idx
                        break
                
                sol[setIdx] = 1
                sol[clrIdx] = 0
                opt = self.opt(sol, C)
                print(f'UnB opt = {opt}')
                assert ((tuple(sol.tolist()) in searched) == False)
                searched.add(tuple(sol.tolist()))
                if opt >= bestOpt:
                    bestOpt = opt
                    bestSol = sol.copy()
                # epilogue
                count += 1
            # sol, lb, ub
            print(f'UnB bestOpt = {bestOpt}')
            # print(f'UnB bestsol = {bestSol}, bestopt = {bestOpt}')
            return bestSol, bestOpt, bestOpt        
        elif policy == 'C2G':
            # force np to reduce GPU memory allocation
            nSelected = 0
            CC = C.reshape((M, M, -1)).cpu().numpy()
            cvg = np.zeros((M, M), dtype=int)
            for i in range(M):
                for j in range(M):
                    for k in range(cvg.shape[0]):
                        cvg[i, j] += np.sum(np.logical_or(CC[i, k], CC[j, k]))
            sol = np.zeros((M,), dtype=bool)
            while N - nSelected >= 2:
                flat = cvg.flatten()
                indices = np.argsort(flat)[::-1] # descending
                i, j = np.unravel_index(indices[0], cvg.shape)
                cvg[i, :] = -1.0
                cvg[j, :] = -1.0
                cvg[:, i] = -1.0
                cvg[:, j] = -1.0
                if i != j:
                    sol[[i, j]] = 1
                    nSelected += 2
                else:
                    sol[i] = 1
                    nSelected += 1
                if "onOptUpdate" in callbacks:
                    callbacks["onOptUpdate"](self.opt(sol, C))
            while nSelected < N:
                lst = [i for i in range(M)]
                diag = cvg[lst, lst]
                idx = np.argmax(diag)
                cvg[idx, idx] = -1
                sol[idx] = 1
                nSelected += 1
                if "onOptUpdate" in callbacks:
                    callbacks["onOptUpdate"](self.opt(sol, C))
            opt = self.opt(sol, C)
            return sol, opt, opt
        elif policy == 'C2I':
            def optMatrixEst(M: int, Q: dict):
                '''
                M: # of cdds
                Q: dict that maps selection of cameras to coverage to a cdd
                    key = (tuple of ints representing selection of camera ids)
                    value = coverage in scalar
                '''
                # gekko solution
                # m = GEKKO(remote=False)
                # B = m.Array(m.Var, (M, M))
                # for camSet in A:
                #     exp = 0
                #     for i in camSet:
                #         exp += B[i, i]
                #     for comb in itertools.combinations(camSet, 2):
                #         exp += (B[comb[0], comb[1]] + B[comb[1], comb[0]])
                #     m.Obj(((exp - A[camSet]) ** 2))
                # # m.options.max_iter = MAX_ITER_GEKKO
                # m.solve(disp=False)
                # return np.array([
                #     [B[i, j] for j in range(M)] for i in range(M)
                # ]).reshape((M, M))
                # solving x argmin |Ax-b|_2
                
                b = []
                A = []
                for camSet in Q:
                    b.append(Q[camSet])
                    s = np.zeros((M, M), dtype=bool)
                    s[list(camSet), :] = 1
                    s[:, list(camSet)] = 1
                    A.append(s.reshape((-1,)).copy())
                b = np.array(b)
                A = np.array(A)
                x = np.linalg.lstsq(A, b, rcond=None)[0]
                return x.reshape((M, M))
            
                
            CC = C.reshape((M, M, -1)).cpu().numpy()
            BB = []
            for i in range(M):
                Q = {}
                # how sel covers i
                for sel in itertools.combinations([k for k in range(M)], 3):
                    s0, s1 ,s2 = sel
                    cvg = np.logical_or(CC[s0, i], CC[s1, i])
                    cvg = np.logical_or(cvg, CC[s2, i])
                    cvg = np.mean(cvg)
                    Q[sel] = cvg
                BB.append(optMatrixEst(M, Q))
            BB = np.array(BB) # M,M,M
            
            # gekko
            BB = np.sum(BB, axis=0) # MxM
            m = GEKKO(remote=False)
            b = m.Array(m.Var, M, integer=True, lb=0, ub=1) # bool select indicator
            # sacrifice precision
            BB = np.around(BB, decimals=3)

            exp = []
            for i in range(M):
                for j in range(M):
                    exp.append(BB[i, j] * b[i] * b[j])
            m.Maximize(m.sum(exp))
            m.options.SOLVER=1  # APOPT is an MINLP solver
            m.Equation(m.sum(b) == N)
            # m.options.solver=1
            # m.options.max_iter = MAX_ITER_GEKKO
            m.solve(disp=False)
            bb = np.array([b[i].value for i in range(b.shape[0])]) # ,dtype=bool in the same line report an error
            bb = np.array(bb, dtype=bool).reshape((-1,))
            assert np.sum(bb) == N
            
            # solve
            # BB = np.sum(BB, axis=0) # MxM
            # # QP https://www.cvxpy.org/examples/basic/quadratic_program.html
            # # MIP https://www.cvxpy.org/examples/basic/mixed_integer_quadratic_program.html
            # x = cp.Variable(M, integer=True)
            # ones = np.ones((M,))
            # prob = cp.Problem(cp.Minimize(-cp.quad_form(x, BB)),
            #     # constraints
            #     [
            #         ones @ x == N,
            #         x >= 0,
            #         x <= 1,
            #     ],
            # )
            # prob.solve()
            # bb = x.value
            
            opt = self.opt(bb, C)
            if "onOptUpdate" in callbacks:
                    callbacks["onOptUpdate"](opt)
            return bb, opt, opt
        elif policy == 'Opt':
            # sol, lb, ub
            sol = np.ones((M,), dtype=bool)
            opt = self.opt(sol, C)
            return sol, opt, opt
        else:
            return NotImplemented