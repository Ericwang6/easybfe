import numpy as np 
import pandas as pd 
import warnings


def maximum_likelihood_estimator(df: pd.DataFrame, bias: float = 0.0):
    """
    Estimate the free energy differences between ligands using a maximum likelihood approach.
    Ref: 10.1021/acs.jcim.9b00528 eq 2-4

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing the following columns:
        - 'ligandA_name': str, name of ligand A in each pairwise comparison.
        - 'ligandB_name': str, name of ligand B in each pairwise comparison.
        - 'ddG.total': float, the free energy difference between ligand B and ligand A.
        - 'ddG_std.total': float, the uncertainty (standard deviation) of the free energy difference.
    bias : float, optional
        A constant bias added to all computed free energy values (default is 0.0).

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns:
        - 'ligand': str, the name of the ligand.
        - 'dG': float, estimated free energy for each ligand.
        - 'dG.std': float, the uncertainty (standard deviation) of the estimated free energy for each ligand.

    Notes
    -----
    The function constructs a linear system from pairwise free energy differences and solves it
    using a maximum likelihood approach to estimate the absolute free energy differences for each ligand.
    The resulting values are computed based on the pseudoinverse of the F matrix, which handles
    overdetermined systems and minimizes numerical instability.

    Examples
    --------
    >>> data = {
    ...     'ligandA_name': ['A', 'A', 'B'],
    ...     'ligandB_name': ['B', 'C', 'C'],
    ...     'ddG.total': [1.2, -0.8, 0.5],
    ...     'ddG_std.total': [0.1, 0.15, 0.2]
    ... }
    >>> df = pd.DataFrame(data)
    >>> maximum_likelihood_estimator(df, bias=0.5)
      ligand   dG   dG.std
    0      A  ...  ...
    1      B  ...  ...
    2      C  ...  ...
    """
    ligand_names = set(df['ligandA_name']).union(df['ligandB_name'])
    names_to_index = {name: idx for idx, name in enumerate(ligand_names)}
    index_to_names = {v: k for k, v in names_to_index.items()}

    data_as_dict = {}
    for _, row in df.iterrows():
        # in the equation, x_ij = x_i - x_j but here in dataframe, ddG is ligandB - ligandA
        edge = (names_to_index[row['ligandB_name']], names_to_index[row['ligandA_name']])
        if edge in data_as_dict:
            warnings.warn(f"Found duplicated perturbation for {row['ligandA_name']}~{row['ligandB_name']}. The first one will be used.")
        data_as_dict[edge] = (row['ddG.total'], row['ddG_std.total'])
        
    # ddG_std(i -> j) = ddG_std((j -> i))
    edges = list(data_as_dict.keys())
    for edge in edges:
        rev_edge = (edge[1], edge[0])
        if rev_edge not in data_as_dict:
            ddG, ddG_std = data_as_dict[edge]
            data_as_dict[rev_edge] = (-ddG, ddG_std)
    
    N = len(ligand_names)
    zVec = np.zeros((N))
    FMat = np.zeros((N, N))
    for edge in data_as_dict:
        i, j = edge
        ddG, ddG_std = data_as_dict[edge]
        zVec[i] += ddG * ddG_std ** (-2)
        FMat[i, j] = -ddG_std ** (-2)
        FMat[i, i] += ddG_std ** (-2)
    
    Finv = np.linalg.pinv(FMat)
    x = np.matmul(Finv, zVec)
    
    res = []
    for i in range(N):
        res.append((index_to_names[i], x[i], Finv[i, i] ** (1/2)))
    res = pd.DataFrame(res, columns=['ligand', 'dG', 'dG_std'])
    res['dG'] += bias
    return res
