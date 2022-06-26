import numpy as np
from scipy import sparse


def cosine_similarity(arr, mutual_length_flag=False):
    """Возвращает попарные значения косинусного расстояния для векторов-строк массива arr.

    Args:
        arr: Входной массив, в котором векторные представления объектов записаны по строками.
        mutual_length_flag: Флаг для генерации дополнительной меры близости 'количество общих координат'
          для возможности разрешения неоднозначностей при равенстве значений косинусных расстояний.

    Returns:
        Structured array с полями 'base' (косинусное расстояние) и 'mutual_length', если
        mutual_length_flag == True, иначе - np.ndarray c косинусными расстояниями.
    """
    A = arr.copy()
    nan_mask = np.isnan(A)
    not_nan_int_mask = (~nan_mask).astype(int)
    # конвертируем not_nan_int_mask в sparse для быстрого умножения
    not_nan_int_mask_sparse = sparse.csr_matrix(not_nan_int_mask)
    A[nan_mask] = 0
    # конвертируем A в sparse для быстрого умножения
    A_sparse = sparse.csr_matrix(A)
    dot_prods = (A_sparse @ A_sparse.T).todense()
    # приведенные нормы векторов-строк:
    # по адресу [i, j] лежит приведенная норма вектора A[i] при умножении его на вектор A[j],
    # т.е. в норму берутся только те компоненты A[i], которые не равны nan в A[j]
    reduced_norms = np.sqrt(A ** 2 @ not_nan_int_mask_sparse.T)
    reduced_norms_prods = reduced_norms * reduced_norms.T
    cosine_sim = np.divide(
        dot_prods, reduced_norms_prods, out=np.zeros_like(dot_prods), where=(reduced_norms_prods != 0)
    )
    if mutual_length_flag:
        # дополнительная мера близости - количество общих координат у векторов-строк исходного массива
        # по адресу [i, j] лежит количество общих (не nan) координат векторов A[i] и A[j]
        mutual_length = not_nan_int_mask @ not_nan_int_mask_sparse.T
        sim = np.empty(shape=cosine_sim.shape, dtype=[("base", np.float32), ("mutual_length", np.int32)])
        sim["base"] = np.round(cosine_sim, decimals=7)
        sim["mutual_length"] = mutual_length
    else:
        sim = np.round(cosine_sim, decimals=7)
    return sim


def pearson_similarity(arr, default_value=0, mutual_length_flag=False):
    """Возвращает попарные значения корреляции Пирсона для векторов-строк массива arr.

    Args:
        arr: Входной массив, в котором векторные представления объектов записаны по строками.
        default_value: Значение близости для векторов, для которых коореляция Пирсона не определена.
        mutual_length_flag: Флаг для генерации дополнительной меры близости 'количество общих координат'
          для возможности разрешения неоднозначностей при равенстве значений коореляции Пирсона.

    Returns:
        Structured array с полями 'base' (коореляция Пирсона) и 'mutual_length', если
        mutual_length_flag == True, иначе - np.ndarray c коореляциями Пирсона.
    """
    A = arr.copy()
    means = np.nanmean(A, axis=1)
    A = A - means.reshape(-1, 1)
    pearson_sim = cosine_similarity(A, mutual_length_flag=True)
    # для векторов-строк A с менее чем двумя общими элементами явно устанавливаем
    # значение корреляции Пирсона равным default_value (корреляция Пирсона не определена,
    # а косинусная близость при этом равна 1)
    pearson_sim["base"][pearson_sim["mutual_length"] < 2] = default_value
    if mutual_length_flag:
        return pearson_sim
    return pearson_sim["base"]
