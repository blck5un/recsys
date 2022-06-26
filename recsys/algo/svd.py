import time

import numpy as np
from tqdm import tqdm

from ..utils import evaluate


class SVD:
    def __init__(self, method="sgd", biased=True):
        """
        Args:
            method: Метод, используемый при разложении ('sgd' или 'als').
            biased: Если True, то при разложении также используются смещения bu и bi.
        """
        self.method = method
        self.biased = biased

    def train(
        self,
        train_dataset,
        n_epochs=10,
        n_factors=10,
        init_mean=0,
        init_std=0.1,
        val_dataset=None,
        check_durations=False,
        verbose=False,
        progress_bar=False,
        random_state=41,
        lr=0.005,
        reg=0.02,
        lr_p=None,
        lr_q=None,
        lr_bu=None,
        lr_bi=None,
        reg_p=None,
        reg_q=None,
        reg_bu=None,
        reg_bi=None,
    ):
        """Выполняет SVD-разложение по заданному при инициализации класса методу.

        Args:
            train_dataset: User-item matrix of ratings in form of unpivot dataframe with columns:
              <user>, <item>, <rating>.
                    n_epochs=10,
            n_epochs: Количество эпох.
            n_factors: Количество факторов - количество столбцов нового представления.
            init_mean: Среднее значение нормального распределения для инициализации.
            init_std: Стандартное отклонение нормального распределения для инициализации.
            val_dataset: Валидационный датасет.
            check_durations: Включение измерения времен различных этапов разложения.
            verbose: Включение логгирования.
            progress_bar: Включение отображения progress bar.
            random_state: Инициализация random state.
            lr: Общий темп обучения (learning rate).
            reg: Общий коэффициент регуляризации.
            lr_p, lr_q, lr_bu, lr_bi: Темпы обучения для соответствующих параметров (p, q, bu, bi),
              если не заданы, то равны lr.
            reg_p, reg_q, reg_bu, reg_bi: Коэффициенты регуляризации для соответствующих параметров
              (p, q, bu, bi), если не заданы, то равны reg.
        """
        np.random.seed(random_state)
        self.fit(train_dataset)
        self.initialize(
            n_factors=n_factors, init_mean=init_mean, init_std=init_std, biased=self.biased,
        )
        evaluate_flag = val_dataset is not None
        results = []

        iterations = range(1, n_epochs + 1)
        if progress_bar:
            iterations = tqdm(iterations)

        for epoch in iterations:
            res = dict()
            res["epoch"] = epoch

            start_time = time.time()
            if self.method == "sgd":
                self.perform_sgd_epoch(
                    lr=lr,
                    reg=reg,
                    lr_p=lr_p,
                    lr_q=lr_q,
                    lr_bu=lr_bu,
                    lr_bi=lr_bi,
                    reg_p=reg_p,
                    reg_q=reg_q,
                    reg_bu=reg_bu,
                    reg_bi=reg_bi,
                )
            else:
                self.perform_als_epoch()
            epoch_time_elapsed = time.time() - start_time

            if verbose:
                print()
                print(f"Epoch: {epoch}. Epoch time elapsed: {epoch_time_elapsed:.2f}")
            if evaluate_flag:
                start_time = time.time()
                train_pred = self.test(train_dataset)
                val_pred = self.test(val_dataset)
                train_metrics = evaluate(train_dataset[self._cols[2]], train_pred, rmse=True, mae=True)
                val_metrics = evaluate(val_dataset[self._cols[2]], val_pred, rmse=True, mae=True)
                res["rmse_train"] = train_metrics["rmse"]
                res["mae_train"] = train_metrics["mae"]
                res["rmse_val"] = val_metrics["rmse"]
                res["mae_val"] = val_metrics["mae"]
                eval_time_elapsed = time.time() - start_time
                if verbose:
                    print(f"Evaluation time elapsed: {eval_time_elapsed:.2f}")
                    print(f'Train: RMSE = {train_metrics["rmse"]:.3f}, MAE = {train_metrics["mae"]:.3f}')
                    print(f'Valid: RMSE = {val_metrics["rmse"]:.3f}, MAE = {val_metrics["mae"]:.3f}')
            if check_durations:
                res["epoch_time_elapsed"] = epoch_time_elapsed
                res["eval_time_elapsed"] = eval_time_elapsed
            results.append(res)

        return results

    def fit(self, train_dataset):
        """Сохраняет train_dataset и формирует вспомогательные структуры данных.

        Args:
            train_dataset: User-item matrix of ratings in form of unpivot dataframe with columns:
              <user>, <item>, <rating>.
        """
        self.train_dataset = train_dataset.copy().reset_index(drop=True)
        self._cols = list(self.train_dataset.columns)
        self._user_col = self._cols[0]
        self._item_col = self._cols[1]
        self._rating_col = self._cols[2]
        users = self.train_dataset.iloc[:, 0].unique()
        items = self.train_dataset.iloc[:, 1].unique()
        # формируем словарь для маппинга users и items в натуральные числа
        self.user_map = {user: i for i, user in enumerate(users)}
        self.item_map = {item: i for i, item in enumerate(items)}
        # self.reverse_user_map = {i: user for user, i in self.user_map.items()}
        # self.reverse_item_map = {i: item for item, i in self.item_map.items()}
        # делаем маппинг user и item
        self.train_dataset[self._user_col] = self.train_dataset[self._user_col].map(self.user_map)
        self.train_dataset[self._item_col] = self.train_dataset[self._item_col].map(self.item_map)
        # запоминаем средний рейтинг по всему трейн-датасету
        self.mean_rating = self.train_dataset[self._rating_col].mean()
        # формируем списки соответствия users/items и index в train_dataset
        self.user_indicies = self.train_dataset.reset_index().groupby(self._user_col)["index"].apply(list)
        self.item_indicies = self.train_dataset.reset_index().groupby(self._item_col)["index"].apply(list)
        self.user_item = self.train_dataset[[self._user_col, self._item_col]].to_numpy(dtype="int32")
        self.ratings = self.train_dataset[self._rating_col].to_numpy(dtype="float32")

    def initialize(self, n_factors, init_mean=0, init_std=0.1, biased=True):
        """Инициализирует p, q, bu, bi случайными значениями из нормального распределения.

        Args:
            n_factors: Количество факторов - количество столбцов нового представления.
            init_mean: Среднее значение нормального распределения.
            init_std: Стандартное отклонение нормального распределения.
            biased: Если True, то при разложении также используются смещения bu и bi.
        """
        n_users = len(self.user_map)
        n_items = len(self.item_map)
        self.p = np.random.randn(n_users, n_factors) * init_std + init_mean
        self.q = np.random.randn(n_items, n_factors) * init_std + init_mean
        if biased:
            self.bu = np.zeros(n_users)
            self.bi = np.zeros(n_items)

    def perform_sgd_epoch(
        self,
        lr=0.005,
        reg=0.02,
        lr_p=None,
        lr_q=None,
        lr_bu=None,
        lr_bi=None,
        reg_p=None,
        reg_q=None,
        reg_bu=None,
        reg_bi=None,
    ):
        """Выполняет одну эпоху SGD.

        Args:
            lr: Общий темп обучения (learning rate).
            reg: Общий коэффициент регуляризации.
            lr_p, lr_q, lr_bu, lr_bi: Темпы обучения для соответствующих параметров (p, q, bu, bi),
              если не заданы, то равны lr.
            reg_p, reg_q, reg_bu, reg_bi: Коэффициенты регуляризации для соответствующих параметров
              (p, q, bu, bi), если не заданы, то равны reg.
        """
        lr_p = lr_p if lr_p is not None else lr
        lr_q = lr_q if lr_q is not None else lr
        lr_bu = lr_bu if lr_bu is not None else lr
        lr_bi = lr_bi if lr_bi is not None else lr
        reg_p = reg_p if reg_p is not None else reg
        reg_q = reg_q if reg_q is not None else reg
        reg_bu = reg_bu if reg_bu is not None else reg
        reg_bi = reg_bi if reg_bi is not None else reg

        indicies = np.random.permutation(self.user_item.shape[0])
        for idx in indicies:
            u, i = self.user_item[idx]
            r_ui = self.ratings[idx]
            pu = self.p[u].copy()
            qi = self.q[i].copy()

            r_ui_est = pu @ qi
            if self.biased:
                r_ui_est += self.mean_rating + self.bu[u] + self.bi[i]
            e_ui = r_ui - r_ui_est
            self.p[u] += lr_p * (e_ui * qi - reg_p * pu)
            self.q[i] += lr_q * (e_ui * pu - reg_q * qi)
            if self.biased:
                self.bu[u] += lr_bu * (e_ui - reg_bu * self.bu[u])
                self.bi[i] += lr_bi * (e_ui - reg_bi * self.bi[i])

    def perform_als_epoch(self, reg=10):
        """Выполняет одну эпоху ALS.

        Args:
            reg: Коэффициент регуляризации.
        """
        for u in self.user_indicies.index:
            qq_sum = 0
            rq_sum = 0
            for idx in self.user_indicies[u]:
                i = self.user_item[idx, 1]
                r_ui = self.ratings[idx]
                qq_sum += np.outer(self.q[i], self.q[i])
                rq_sum += r_ui * self.q[i]
            self.p[u] = np.linalg.inv(reg * np.identity(qq_sum.shape[0]) + qq_sum) @ rq_sum

        for i in self.item_indicies.index:
            pp_sum = 0
            rp_sum = 0
            for idx in self.item_indicies[i]:
                u = self.user_item[idx, 0]
                r_ui = self.ratings[idx]
                pp_sum += np.outer(self.p[u], self.p[u])
                rp_sum += r_ui * self.p[u]
            self.q[i] = np.linalg.inv(reg * np.identity(pp_sum.shape[0]) + pp_sum) @ rp_sum

    def test(self, test_dataset):
        """Формирует предсказания для test_dataset.

        Args:
            test_dataset: User-item matrix of ratings in form of unpivot dataframe with columns:
              <user>, <item>, <rating>.

        Returns:
            Numpy array c рейтингами для каждой строки входного датафрейма, порядок элементов сохраняется.
        """
        if list(test_dataset.columns) != self._cols:
            raise ValueError("Столбцы test_dataset отличаются от столбцов train_dataset!")

        df = test_dataset.copy()
        df[self._user_col] = df[self._user_col].map(self.user_map).fillna(-1)
        df[self._item_col] = df[self._item_col].map(self.item_map).fillna(-1)
        test_user_item = df[[self._user_col, self._item_col]].to_numpy(dtype="int32")

        test_ratings = []
        for u, i in test_user_item:
            # если данного user или item нет в трейне, то присваиваем рейтингу среднее значение по трейну
            if u == -1 or i == -1:
                rating = self.mean_rating
            else:
                rating = self.p[u] @ self.q[i]
                if self.method == "sgd" and self.biased:
                    rating += self.mean_rating + self.bu[u] + self.bi[i]
            test_ratings.append(rating)
        return np.array(test_ratings)
