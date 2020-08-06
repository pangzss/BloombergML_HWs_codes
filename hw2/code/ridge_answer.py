def main():
	lasso_data_fname = "lasso_data.pickle"
	x_train, y_train, x_val, y_val, target_fn, coefs_true, featurize = load_problem(lasso_data_fname)

	# Generate features
	X_train = featurize(x_train)
	X_val = featurize(x_val)

	#Visualize training data
	fig, ax = plt.subplots()
	ax.imshow(X_train)
	ax.set_title("Design Matrix: Color is Feature Value")
	ax.set_xlabel("Feature Index")
	ax.set_ylabel("Example Number")
	plt.show(block=False)

	# Compare our RidgeRegression to sklearn's.
	compare_our_ridge_with_sklearn(X_train, y_train, l2_reg = 1.5)

	# Do hyperparameter tuning with our ridge regression
	grid, results = do_grid_search_ridge(X_train, y_train, X_val, y_val)
	print(results)

	# Plot validation performance vs regularization parameter
	fig, ax = plt.subplots()
#    ax.loglog(results["param_l2reg"], results["mean_test_score"])
	ax.semilogx(results["param_l2reg"], results["mean_test_score"])
	ax.grid()
	ax.set_title("Validation Performance vs L2 Regularization")
	ax.set_xlabel("L2-Penalty Regularization Parameter")
	ax.set_ylabel("Mean Squared Error")
	fig.show()

	# Let's plot prediction functions and compare coefficients for several fits
	# and the target function.
	pred_fns = []
	x = np.sort(np.concatenate([np.arange(0,1,.001), x_train]))
	name = "Target Parameter Values (i.e. Bayes Optimal)"
	pred_fns.append({"name":name, "coefs":coefs_true, "preds": target_fn(x) })

	l2regs = [0, grid.best_params_['l2reg'], 1]
	X = featurize(x)
	for l2reg in l2regs:
		ridge_regression_estimator = RidgeRegression(l2reg=l2reg)
		ridge_regression_estimator.fit(X_train, y_train)
		name = "Ridge with L2Reg="+str(l2reg)
		pred_fns.append({"name":name,
						 "coefs":ridge_regression_estimator.w_,
						 "preds": ridge_regression_estimator.predict(X) })

	f = plot_prediction_functions(x, pred_fns, x_train, y_train, legend_loc="best")
	f.show()

	f = compare_parameter_vectors(pred_fns)
	f.show()


	##Sample code for plotting a matrix
	## Note that this is a generic code for confusion matrix
	## You still have to make y_true and y_pred by thresholding as per the insturctions in the question.
	y_true = [1, 0, 1, 1, 0, 1]
	y_pred = [0, 0, 1, 1, 0, 1]
	eps = 1e-1;
	cnf_matrix = confusion_matrix(y_true, y_pred)
	plt.figure()
	plot_confusion_matrix(cnf_matrix, title="Confusion Matrix for $\epsilon = {}$".format(eps), classes=["Zero", "Non-Zero"])
	plt.show()