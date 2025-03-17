import numpy as np

def feature_search_demo(data):
    current_set_of_features = []
    best_feature = []
    best_accuracy_total = 0.0
    print("Beggining search")
    for i in range(1, data.shape[1]):
        print(f"On the {i}th level of the search tree")
        feature_to_add_at_this_level = None
        best_accuracy_so_far = 0
        for k in range(1, data.shape[1]):
            if (k not in current_set_of_features):
                print(f'--Considering adding the {k} feature')
                accuracy = leave_one_out_cross_validation(data, current_set_of_features, k)
                print(f'----Accuracy with feature {k}: {(accuracy * 100):.1f}%')
                if accuracy > best_accuracy_so_far:
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = k
        if feature_to_add_at_this_level is not None:
            current_set_of_features.append(feature_to_add_at_this_level)
            print(f"On level {i} i added feature {feature_to_add_at_this_level} to current set")
            if best_accuracy_so_far > best_accuracy_total:
                best_accuracy_total = best_accuracy_so_far
                best_feature = list(current_set_of_features)
        print(f"Best feature subset found: {best_feature} with accuracy: {(best_accuracy_total * 100):.1f}")



def leave_one_out_cross_validation(data, current_set, feature_to_add):
    number_correctly_classified = 0
    selected_features = list(current_set) + [feature_to_add]
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
    data = np.loadtxt('CS170_Small_Data__1.txt')
    feature_search_demo(data)
    # leave_one_out_cross_validation(data)

main()