import numpy as np

from .. import similarities as sims


class KNNBasic:
    def __init__(self):
        self._with_means = False

    def train(self, train_dataset, rating_scale=(1, 5), user_based=True, similarity="cosine"):
        """Сохраняет train_dataset и формирует вспомогательные структуры данных.

        Args:
            train_dataset: User-item matrix of ratings in form of unpivot dataframe with columns:
              <user>, <item>, <rating>.
            rating_scale: Диапазон значений рейтинга.
            user_based: True - использовать user-based подход, False - использовать item-based подход.
            similarity: Мера близости ('cosine', 'pearson').
        """
        self.train_dataset = train_dataset.copy().reset_index(drop=True)
        self.rating_scale = rating_scale
        self.user_based = user_based
        self.similarity = similarity
        self._cols = list(self.train_dataset.columns)
        self._user_col = self._cols[0]
        self._item_col = self._cols[1]
        self._rating_col = self._rating_col
        users = self.train_dataset.iloc[:, 0].unique()
        items = self.train_dataset.iloc[:, 1].unique()
        # формируем словарь для маппинга users и items в натуральные числа
        self.user_map = {user: i for i, user in enumerate(users)}
        self.item_map = {item: i for i, item in enumerate(items)}
        # self.reverse_user_map = {i: user for user, i in self.user_map.items()}
        # self.reverse_item_map = {i: item for item, i in self.item_map.items()}
        self.train_dataset[self._user_col] = self.train_dataset[self._user_col].map(self.user_map)
        self.train_dataset[self._item_col] = self.train_dataset[self._item_col].map(self.item_map)
        self.mean_rating = self.train_dataset[self._rating_col].mean()
        # формируем user-item матрицу из трейн-датасета
        self.train_user_item_matrix = self.train_dataset.pivot(
            index=self._user_col, columns=self._item_col, values=self._rating_col
        ).to_numpy()
        # для того, чтобы выполнить item-based подход, достаточно транспонировать матрицу
        if not self.user_based:
            self.train_user_item_matrix = self.train_user_item_matrix.T
        self.not_nan_mask = ~np.isnan(self.train_user_item_matrix)
        self.train_user_item_matrix[~self.not_nan_mask] = 0
        # считаем попарные значения меры близости для векторов-строк user-item матрицы
        if self.similarity == "cosine":
            self.sim = sims.cosine_similarity(self.train_user_item_matrix, mutual_length_flag=True)
        else:
            self.sim = sims.pearson_similarity(self.train_user_item_matrix, mutual_length_flag=True)
        # заполняем нулями диагональ, чтобы не учитывать себя при поиске ближайшего соседа
        np.fill_diagonal(self.sim["base"], 0)
        # обнуляем отрицательные значения, чтобы не учитывать их при поиске ближайшего соседа
        self.sim["base"][self.sim["base"] < 0] = 0
        if self._with_means:
            self.user_mean = self.train_dataset.groupby(self._user_col)[self._rating_col].mean().to_numpy()
            self.item_mean = self.train_dataset.groupby(self._item_col)[self._rating_col].mean().to_numpy()

    def test(self, test_dataset, k_neighbors=40):
        """Считает рейтинги для test_dataset.

        Args:
            test_dataset: User-item matrix of ratings in form of unpivot dataframe with columns:
              <user>, <item>, <rating>.
            k_neighbors: Количество ближайших соседей, учитываемых при расчете рейтинга.

        Returns:
            Numpy array c рейтингами для каждой строки входного датафрейма, порядок элементов сохраняется.
        """
        if list(test_dataset.columns) != self._cols:
            raise ValueError("Столбцы test_dataset отличаются от столбцов train_dataset!")

        df = test_dataset.copy()
        df[self._user_col] = df[self._user_col].map(self.user_map).fillna(-1).astype("int32")
        df[self._item_col] = df[self._item_col].map(self.item_map).fillna(-1).astype("int32")
        if self.user_based:
            test_user_item = df[[self._user_col, self._item_col]].to_numpy(dtype="int32")
            if self._with_means:
                train_means = self.user_mean
        else:
            test_user_item = df[[self._item_col, self._user_col]].to_numpy(dtype="int32")
            if self._with_means:
                train_means = self.item_mean
        test_ratings = []
        # работаем как будто с user-item (хотя при item-based тут будет item-user)
        for u, i in test_user_item:
            # если данного user или item нет в трейне, то присваиваем рейтингу среднее значение по трейну
            if u == -1 or i == -1:
                rating_est = self.mean_rating
            else:
                users_idx = np.where(self.not_nan_mask[:, i])[0]
                k = min(k_neighbors, len(users_idx))
                neighbors_idx = np.argpartition(
                    self.sim[u, users_idx], -k, order=["base", "mutual_length"]
                )[-k:]
                neighbors_sim = self.sim["base"][u, users_idx[neighbors_idx]]
                neighbors_ratings = self.train_user_item_matrix[users_idx[neighbors_idx], i]
                if self._with_means:
                    neighbors_means = train_means[users_idx[neighbors_idx]]
                    neighbors_ratings -= neighbors_means
                # считаем взвешенную сумму рейтингов (веса - значения близости)
                weighted_ratings_sum = (neighbors_sim * neighbors_ratings).sum()
                weights_sum = neighbors_sim.sum()
                # определяем надо ли подставить default_value
                default_value_flag = np.isclose(weights_sum, 0, atol=0.0001)
                if self.similarity == "pearson":
                    # максимальное количество общих координат у соседей
                    neighbors_max_freq = self.sim["mutual_length"][u, users_idx[neighbors_idx]].max()
                    default_value_flag |= neighbors_max_freq < 2
                if default_value_flag:
                    rating_est = self.mean_rating
                else:
                    rating_est = weighted_ratings_sum / weights_sum
                    if self._with_means:
                        rating_est += train_means[u]
            test_ratings.append(rating_est)

        test_ratings = np.clip(test_ratings, self.rating_scale[0], self.rating_scale[1])
        return test_ratings


class KNNWithMeans(KNNBasic):
    def __init__(self):
        super().__init__()
        self._with_means = True
