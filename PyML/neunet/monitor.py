def monitor(network,
        training_data, evaluation_data,
        training_cost,training_accuracy,
        evaluation_cost, evaluation_accuracy,
        lmbda,
        monitor_evaluation_cost=False,
        monitor_evaluation_accuracy=False,
        monitor_training_cost=False,
        monitor_training_accuracy=False):

    n = len(training_data)
    if evaluation_data: n_vdata = len(evaluation_data)

    if monitor_training_cost:
        cost = network.total_cost(training_data, lmbda)
        training_cost.append(cost)

    if monitor_training_accuracy:
        accuracy = network.accuracy(training_data, convert=True)
        training_accuracy.append(accuracy*1.0 / n)

    if monitor_evaluation_cost:
        cost = network.total_cost(evaluation_data, lmbda)
        evaluation_cost.append(cost)

    if monitor_evaluation_accuracy:
        accuracy = network.accuracy(evaluation_data, convert=True)
        print accuracy
        evaluation_accuracy.append(accuracy*1.0 / n_vdata)

    pass