import time

from tsfresh import extract_features
import numpy as np

def ts_features(train1, column_sort, column_value, settings):
    X = extract_features(train1,
                         column_id=KEY_NAME,
                         column_sort=column_sort,
                         column_value=column_value,
                         default_fc_parameters=settings,
                         impute_function=impute,
                         disable_progressbar=True,
                         show_warnings=True)
    return X


def generate_ts_features(X_train, y_col_name, time_col, settings):
    start = time.time()
    features = ts_features(X_train, time_col, y_col_name, settings)
    features = reduce_mem_usage(features)
    X_filtered = features.replace([np.inf, -np.inf, np.nan], 0)

    end = time.time()
    print("Time", end - start)

    ####
    ###

    return X_filtered


def generate_and_merge(X_train, X_test, settings):
    features = generate_ts_features(X_train, 'meter_reading_log1p', 'hour_datetime', settings)

    X_train = X_train.merge(features, how='left', left_on=KEY_NAME, right_on='id')
    X_test = X_test.merge(features, how='left', left_on=KEY_NAME, right_on='id')

    return X_train, X_test


folds = 6


@telegram_sender(token=TELEGRAM_API_KEY, chat_id=CHAT_ID)
def lgbm_cross_validation(params):
    seed = 42

    kf = KFold(n_splits=folds)
    total_loss = []

    for train_index, val_index in kf.split(full_train_df):
        train_X = full_train_df.loc[train_index, [*feat_cols, 'meter_reading_log1p']].reset_index(drop=True)
        val_X = full_train_df.loc[val_index, feat_cols].reset_index(drop=True)
        train_y = target.iloc[train_index]
        val_y = target.iloc[val_index]

        ###

        train_X, val_X = generate_and_merge(train_X, val_X, settings)

        ###
        train_X.drop('meter_reading_log1p', axis=1, inplace=True)
        ####

        ####

        lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categoricals)
        lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categoricals)
        lgbm = lgb.train(params,
                         lgb_train,
                         num_boost_round=500,
                         valid_sets=(lgb_train, lgb_eval),
                         early_stopping_rounds=100,
                         verbose_eval=0)

        pred_y = lgbm.predict(val_X)
        mse = np.sqrt(mean_squared_error((val_y), (pred_y)))
        total_loss.append(mse)

    return {'loss': np.mean(total_loss), 'status': STATUS_OK, 'params': params}


def optimize_lgbm(max_evals=1000):
    space = {
        'metric': {'rmse'},
        'num_leaves': scope.int(hp.quniform('num_leaves', 30, 150, 1)),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        'min_data_in_leaf': scope.int(hp.qloguniform('min_data_in_leaf', 0, 6, 1)),
        'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
        'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
        'min_child_weight': hp.loguniform('min_child_weight', -16, 5),  # also aliases to min_sum_hessian
        'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
        'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
        'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    }

    trials = Trials()
    best = fmin(fn=lgbm_cross_validation,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                verbose=1)

    # find the trial with lowest loss value. this is what we consider the best one
    idx = np.argmin(trials.losses())
    print(idx)

    print(trials.trials[idx])
    # these should be the training parameters to use to achieve the best score in best trial
    params = trials.trials[idx]["result"]["params"]

    print(params)
    return params


def generate_and_merge(X_train, X_test, settings_week, settings_hour, kfold_settings):
    features_1 = generate_ts_features(X_train, 'meter_reading_log1p', 'dayofyear_datetime', settings_week)
    features_2 = generate_ts_features(X_train, 'meter_reading_log1p', 'hour_datetime', settings_hour)

    X_train = X_train.merge(pd.concat((features_1, features_2), axis=1), how='left', left_on=KEY_NAME, right_on='id')
    kfold = KFold(2)
    features_part = pd.DataFrame()
    for i, (train_index, test_index) in enumerate(kfold.split(X_train)):
        train_part = X_train.loc[test_index]
        features_1 = generate_ts_features(
            train_part, 'dew_temperature', 'dayofyear_datetime', kfold_settings, 'building_id').add_suffix("_" + str(i))
        features_2 = generate_ts_features(
            train_part, 'sea_level_pressure', 'dayofyear_datetime', kfold_settings, 'building_id').add_suffix(
            "_" + str(i))

        features_3 = generate_ts_features(
            train_part, 'sea_level_pressure', 'dayofyear_datetime', kfold_settings, 'building_id').add_suffix(
            "_" + str(i))

        f = pd.concat((features_1, features_2, features_3), axis=1)
        features_part = pd.concat((features_part, f), axis=1)
        print(features_part.shape)

    features_1 = generate_ts_features(X_train, 'sea_level_pressure', 'weekofyear_datetime', kfold_settings)
    features_2 = generate_ts_features(X_train, 'dew_temperature', 'weekofyear_datetime', kfold_settings)
    features_3 = generate_ts_features(X_train, 'air_temperature', 'weekofyear_datetime', kfold_settings)

    features = pd.concat((features_part, features_1, features_2, features_3), axis=1)
    X_train = X_train.merge(features, how='left', left_on=KEY_NAME, right_on='id')

    if np.any(X_test):
        X_test = X_test.merge(features, how='left', left_on=KEY_NAME, right_on='id')

    return X_train, X_test, features