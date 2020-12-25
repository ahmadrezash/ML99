from p3.models import RidgeModel, PolynomialModel, LassoModel, RidgeCVModel

if __name__ == '__main__':

    # Part آ (use ModelA)

    ####### Data 1 #######
    my_model = PolynomialModel(
        data_path="./data1.csv"
    )
    for i in range(1, 10):
        my_model.build_run_eval_model(M=i)

    my_model.plot_acc_history()
    my_model.plot_per_eval_history()

    # ####### Data 2 #######
    my_model = PolynomialModel(
        data_path="./data2.csv"
    )
    for i in range(1, 10):
        my_model.build_run_eval_model(M=i)

    my_model.plot_acc_history()
    my_model.plot_per_eval_history()

    # Part ب (use ModelA)

    ####### Data 1 #######
    my_model = PolynomialModel(
        train_data_path="./data1.csv",
        test_data_path="./data3.csv"
    )
    for i in range(1, 10):
        my_model.build_run_eval_model(M=i)

    my_model.plot_acc_history()
    my_model.plot_per_eval_history()

    my_model.plot_condition_num_history()

    # ####### Data 2 #######
    my_model = PolynomialModel(
        train_data_path="./data1.csv",
        test_data_path="./data3.csv"
    )
    for i in range(1, 10):
        my_model.build_run_eval_model(M=i)

    my_model.plot_acc_history()
    my_model.plot_per_eval_history()

    my_model.plot_condition_num_history()
    # Part د
    ####### Data 2 -L1 #######
    my_model = RidgeModel(
        train_data_path="./data1.csv",
        test_data_path="./data3.csv"
    )
    start = 10 ** (-7)
    stop = 10 ** (-5)
    step = 10 ** (-7)
    import numpy as np

    for i in np.arange(start, stop, step):
        my_model.build_run_eval_model(M=9, alpha=i)

    my_model.plot_acc_history()
    my_model.plot_per_eval_history()

    ####### Data 2 - L2 #######
    my_model = LassoModel(
        train_data_path="./data1.csv",
        test_data_path="./data3.csv"
    )
    start = 10 ** (-7)
    stop = 10 ** (-5)
    step = 10 ** (-7)
    import numpy as np

    for i in np.arange(start, stop, step):
        my_model.build_run_eval_model(M=9, alpha=i)

    my_model.plot_acc_history()
    my_model.plot_per_eval_history()

    # Part هـ
    my_model.plot_w_and_error()

    ## Part و
    my_model = RidgeCVModel(
        train_data_path="./data1.csv",
        test_data_path="./data3.csv"
    )
    start = 10 ** (-7)
    stop = 10 ** (-5)
    step = 10 ** (-7)
    import numpy as np

    # for i in np.arange(start, stop, step):
    my_model.build_run_eval_model(M=9, alpha=np.arange(start, stop, step))
    my_model.plot_w_and_error()
    my_model.plot_acc_history()
    my_model.plot_per_eval_history()
