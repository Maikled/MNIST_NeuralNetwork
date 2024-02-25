import matplotlib.pyplot as plt
import numpy
import fileManager
import neuralNetwork
import stats

input_nodes = 784
hidden_layers = 1
hidden_layer_nodes = 200
output_nodes = 10
learning_rate = 0.1
epochs = 10

neuralNetworkObject = neuralNetwork.neuralNetwork(input_nodes, output_nodes, hidden_layers, hidden_layer_nodes, learning_rate)

file_manager = fileManager.Manager('NN.txt')
if not file_manager.exist_file():
    training_data_list = fileManager.Manager.load_mnist_train_data('mnist_dataset/mnist_train.csv')
    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            neuralNetworkObject.train(inputs, targets)
        print(f'Number of end epochs: {e + 1}')

    list_for_write = [[input_nodes, output_nodes, hidden_layers, hidden_layer_nodes, learning_rate]]
    for matrix in neuralNetworkObject.matrixs:
        list_for_write.append(matrix)

    file_manager.save_data_pickle(list_for_write)
else:
    data = file_manager.load_data_pickle()
    input_nodes, output_nodes, hidden_layers, hidden_layer_nodes, learning_rate = data[0]
    neuralNetworkObject = neuralNetwork.neuralNetwork(input_nodes, output_nodes, hidden_layers, hidden_layer_nodes, learning_rate)
    neuralNetworkObject.matrixs = data[1:]

#test_dataset = fileManager.Manager.load_images_test_data('my_own_images')
test_dataset = fileManager.Manager.load_mnist_test_data('mnist_dataset/mnist_test.csv')

show_image = False
results = list()
for test_data in test_dataset:
    if show_image:
        plt.imshow(test_data[1].reshape(28, 28), cmap='Greys')
        plt.show()

    outputs = neuralNetworkObject.query(test_data[1])
    neural_answer = numpy.argmax(outputs)
    answer = test_data[0]

    results.append([answer, neural_answer, outputs])
    label = f'network answer: {neural_answer}, right answer: {answer}, '
    if neural_answer == answer:
        label += 'PASS'
    else:
        label += 'FAILED'
    print(label)

efficiency = stats.get_network_efficiency(results)
print(F'\nЭффективность = {efficiency}')

stats_file_path = 'my_stats/NN_stats.txt'
stats.add_stats_to_file([[input_nodes, hidden_layers, hidden_layer_nodes, output_nodes, learning_rate, epochs, efficiency]], stats_file_path)
