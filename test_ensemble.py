from utils import *

if __name__ == "__main__":
    try:
        model_num = 2
        model_name = 'xgb'
        x_train, y_train, test = load_data()
        x_train_res, y_train_res = smote_data(x_train, y_train, 0.5)
        predict_test = run_ensemble(x_train_res, y_train_res, x_train, y_train, test, model_num)
        mean_y_pred = np.array(predict_test).mean(axis=0)
        pd.DataFrame(mean_y_pred).set_index(test.index).to_csv('ensemble_nni_'+model_name+'.csv')

    except Exception as exception:
        logger.exception(exception)
        raise