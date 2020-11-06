import pandas as pd
import numpy as np

def select_for_metric(df_answer, df_submission):
    '''
    Args | df_answer: Pandas DataFrame | df_submission: Pandas DataFrame
    Return | true: Numpy Array | pred: Numpy Array
    '''
    df_1 = df_answer
    df_2 = df_submission

    id_column = df_1.columns[0]

    df_1.index = df_1[id_column]

    df_2.index = df_2[id_column]
    df_2 = df_2.loc[df_1.index]
    
    return df_1, df_2

def gap(true_df, pred_df):
    true_df, pred_df = select_for_metric(true_df, pred_df)
    """
    Compute Global Average Precision score (GAP)
    Parameters
    ----------
    y_true : Dict[Any, Any]
        Dictionary with query ids and true ids for query samples
    y_pred : Dict[Any, Tuple[Any, float]]
        Dictionary with query ids and predictions (predicted id, confidence
        level)
    Returns
    -------
    float
        GAP score
    Examples
    --------
    >>> y_true = {
    ...         'id_001': 123,
    ...         'id_002': 123,
    ...         'id_003': 999,
    ...         'id_004': 123,
    ...         'id_005': 999,
    ...         'id_006': 888,
    ...         'id_007': 666,
    ...         'id_008': 666,
    ...         'id_009': 123,
    ...         'id_010': 666,
    ...     }
    >>> y_pred = {
    ...         'id_001': (123, 0.15),
    ...         'id_002': (123, 0.10),
    ...         'id_003': (999, 0.30),
    ...         'id_005': (999, 0.40),
    ...         'id_007': (555, 0.60),
    ...         'id_008': (666, 0.70),
    ...         'id_010': (666, 0.99),
    ...     }
    >>> gap(y_true, y_pred)
    0.5479166666666666
    >>> it’s 1 if the i-th prediction is correct, and 0 otherwise
    """
    y_pred = {}
    for i, value in zip(pred_df['id'], pred_df[['landmark_id', 'conf']].values):
        y_pred[i] = tuple(value)
        
    y_true = {}
    for i, value in zip(true_df['id'], true_df[['landmark_id']].values):
        y_true[i] = tuple(value)
        
    indexes = list(y_pred.keys())
    indexes.sort(
        key=lambda x: -y_pred[x][1],
    )
    queries_with_target = len([i for i in y_true.values() if i is not None])
    correct_predictions = 0
    total_score = 0.
    for i, k in enumerate(indexes, 1):
        relevance_of_prediction_i = 0
        if y_true[k] == y_pred[k][0]:
            correct_predictions += 1
            relevance_of_prediction_i = 1
        precision_at_rank_i = correct_predictions / i
        total_score += precision_at_rank_i * relevance_of_prediction_i

    return 1 / queries_with_target * total_score
