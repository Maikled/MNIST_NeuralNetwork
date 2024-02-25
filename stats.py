import fileManager


def get_network_efficiency(results):
    right_answers = [result for result in results if result[0] == result[1]]
    efficiency = len(right_answers) / len(results)
    return efficiency


def get_network_precision(results):
    right_answers = [result for result in results if result[0] == result[1]]
    wrong_answers = [result for result in results if result[0] != result[1]]
    precision = len(right_answers) / (len(right_answers) + len(wrong_answers))
    return precision


def add_stats_to_file(stats, path):
    file_manager = fileManager.Manager(path)
    if file_manager.exist_file():
        load_stats_list = list(file_manager.load_data_json())
        load_stats_list.append(stats[0])
        file_manager.save_data_json(load_stats_list)
    else:
        file_manager.save_data_json(stats)
