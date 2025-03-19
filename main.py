import numpy as np
import time

def forward_selection(data):
    start_time = time.time()
    current_set_of_features = []
    best_feature = []
    best_accuracy_total = 0.0
    print("Beginning search")
    for i in range(1, data.shape[1]):
        feature_to_add_at_this_level = None
        best_accuracy_so_far = 0
        for k in range(1, data.shape[1]):
            if (k not in current_set_of_features):
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k)
                print(f"  Using feature(s) {{{', '.join(map(str, current_set_of_features + [k]))}}} accuracy is {(accuracy * 100):.1f}%")                
                if accuracy > best_accuracy_so_far:
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = k
        if feature_to_add_at_this_level is not None:
            current_set_of_features.append(feature_to_add_at_this_level)
            if best_accuracy_so_far > best_accuracy_total:
                best_accuracy_total = best_accuracy_so_far
                best_feature = list(current_set_of_features)
        if (best_accuracy_so_far < best_accuracy_total):
            print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
        print(f"Feature set {best_feature} was best, accuracy is {(best_accuracy_total * 100):.1f}%")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Finished search!! The best feature subset is {best_feature}, which has an accuracy of {(best_accuracy_total*100):.1f}%")
    print(f"Forward Selection completed in {elapsed_time / 3600:.2f} hours.")


def backward_selection(data):
    start_time = time.time()
    current_set_of_features = list(range(1,data.shape[1]))
    best_feature = []
    best_accuracy_total = 0.0
    print("Beginning search")
    while len(current_set_of_features) > 0:
        feature_to_remove_at_this_level = None
        best_accuracy_so_far = 0
        for k in current_set_of_features:
            remaining_features = [f for f in current_set_of_features if f != k]
            accuracy = leave_one_out_cross_validation(data, remaining_features, None)
            print(f"  Using feature(s) {{{', '.join(map(str, remaining_features))}}} accuracy is {(accuracy * 100):.1f}%")
            if accuracy > best_accuracy_so_far:
                best_accuracy_so_far = accuracy
                feature_to_remove_at_this_level = k
        if feature_to_remove_at_this_level is not None:
            current_set_of_features.remove(feature_to_remove_at_this_level)
            if best_accuracy_so_far > best_accuracy_total:
                best_accuracy_total = best_accuracy_so_far
                best_feature = list(current_set_of_features)
        if best_accuracy_so_far < best_accuracy_total:
            print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
        print(f"Feature set {best_feature} was best, accuracy is {(best_accuracy_total * 100):.1f}%")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Finished search!! The best feature subset is {best_feature}, which has an accuracy of {(best_accuracy_total*100):.1f}%")
    print(f"Backward Elimination completed in {elapsed_time / 3600:.2f} hours.")


def leave_one_out_cross_validation(data, current_set, feature_to_add):
    number_correctly_classified = 0
    selected_features = list(current_set)
    if feature_to_add is not None:
        selected_features.append(feature_to_add)
    masked_data = np.zeros_like(data)
    masked_data[:, 0] = data[:, 0]
    masked_data[:, selected_features] = data[:, selected_features]
    for i in range(len(masked_data)):
        object_to_classify = masked_data[i, 1:]
        label_object_to_classify = masked_data[i, 0]
        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')
        for k in range(len(masked_data)):
            if k != i:
                distance = np.sqrt(np.sum((object_to_classify - masked_data[k, 1:])**2))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = masked_data[nearest_neighbor_location, 0]
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
    accuracy = number_correctly_classified / len(masked_data)
    return accuracy



def main():
    validMenuOption = False
    while (validMenuOption != True):
        print("Welcome to Simon Chao's Feature Selection Algorithm")
        data_name = input("Type in the name of the file to test\n")
        data = np.loadtxt(data_name)
        data_feature = data.shape[1] - 1
        data_instance = data.shape[0]
        feature_indices = list(range(1,data.shape[1]))
        nearest_neighbor_accuracy = leave_one_out_cross_validation(data,feature_indices, None)
        validNumOption = False
        while (validNumOption != True):
            print("Type the number of the algorithm you want to run.")
            print("1. Forward Selection")
            print("2. Backward Elimination") 
            option = input()
            if option == "1":
                print(f"This dataset has {data_feature} features (not including the class attribute), with {data_instance} instances.")
                print(f"Running nearest neighbor with all {data_feature} features, using leaving-one-out evaluation, I get an accuracy of {(nearest_neighbor_accuracy*100):.1f}%")
                forward_selection(data)
                validMenuOption = True
                validNumOption = True
            elif option == '2':
                print(f"This dataset has {data_feature} features (not including the class attribute), with {data_instance} instances.")
                print(f"Running nearest neighbor with all {data_feature} features, using leaving-one-out evaluation, I get an accuracy of {(nearest_neighbor_accuracy*100):.1f}%")
                backward_selection(data)
                validMenuOption = True
                validNumOption = True
            else:
                print("Please select a valid option")

main()