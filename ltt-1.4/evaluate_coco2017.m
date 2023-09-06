[dtr, dte, t] = create_data(16, 2);

bin_g.model = load('results/coco2017_bin_20-Jun-2023.mat').t_hat;

bin_g.ll_train = forrest_ll_fast(dtr, bin_g.model);
bin_g.ll_test = forrest_ll_fast(dte, bin_g.model);

bin_g.nll_train_avg = -bin_g.ll_train / size(dtr.x, 2);
bin_g.nll_test_avg = -bin_g.ll_test / size(dte.x, 2);

bin_g.bic_train = -bic(dtr, bin_g.model);
bin_g.bic_test = -bic(dte, bin_g.model);

cl.model = load('results/coco2017_cl_20-Jun-2023.mat').t_hat;

cl.ll_train = forrest_ll_fast(dtr, cl.model);
cl.ll_test = forrest_ll_fast(dte, cl.model);

cl.nll_train_avg = -cl.ll_train / size(dtr.x, 2);
cl.nll_test_avg = -cl.ll_test / size(dte.x, 2);

cl.bic_train = -bic(dtr, cl.model);
cl.bic_test = -bic(dte, cl.model);
