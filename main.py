import numpy as np
import math

def feature_search_demo(data):
    current_set_of_features = []
    for i in range(1, len(data[0])):
        print(f"On the {i}th level of the search tree")
        feature_to_add_at_this_level = []
        best_accuracy_so_far = 0
        for k in range(1, len(data[0])):
            if (k not in current_set_of_features):
                print(f'--Considering adding the {k} feature')
                # accuracy = leave_one_out_cross_validation(data, current_set_of_features, k + 1)
            accuracy = 0
            if accuracy > best_accuracy_so_far:
                best_accuracy_so_far = accuracy
                feature_to_add_at_this_level = k
        print(f"On level {i} i added feature {feature_to_add_at_this_level} to current set")


def leave_one_out_cross_validation(data):
    for i in range(len(data)):
        object_to_classify = data[i, 1:]
        label_object_to_classify = data[i, 0]
        # print(f"Looping over i, at the {i + 1} location")
        # print(f"The {i + 1}th object is in class {label_object_to_classify}")
        for k in range(len(data)):
            if k != i:
                distance = math.sqrt()
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = nearest_neighbor_location
                # print(f'Ask if {i} is nearest neigbour with {k}')




def main():
    data = np.loadtxt('CS170_Small_Data__1.txt')
    # feature_search_demo(data)
    leave_one_out_cross_validation(data)

main()



