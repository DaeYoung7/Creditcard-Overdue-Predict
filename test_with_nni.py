from utils import *

if __name__ == "__main__":
    try:
        model_num = int(sys.argv[1])
        x_train, y_train, test = load_data()
        x_train_res, y_train_res = smote_data(x_train, y_train, 0.5)

        # RECEIVED_PARAMS = {'max_leaves':20, 'learning_rate':0.1, 'depth':10, 'border_count':100,\
        #                    'l2_leaf_reg':0.1, 'iterations':200}
        RECEIVED_PARAMS = nni.get_next_parameter()
        run(x_train_res, y_train_res, x_train, y_train, RECEIVED_PARAMS, model_num)
        # mean_y_pred = np.array(bagging_predict_result).mean(axis=0)
        # print(f'log loss: {log_loss(y_train, mean_y_pred)}')

    except Exception as exception:
        logger.exception(exception)
        raise