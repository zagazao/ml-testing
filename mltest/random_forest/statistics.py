from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from mltest.random_forest.base import RFTest


class PropertyTest(RFTest):

    def run_test(self, sk_model_obj, x_test, y_test):

        node_counts = [model.tree_.node_count for model in sk_model_obj.estimators_]
        leaf_counts = [model.tree_.n_leaves for model in sk_model_obj.estimators_]
        forest_depths = [model.tree_.max_depth for model in sk_model_obj.estimators_]

        feature_importance = sk_model_obj.feature_importances_

        forest_num_nodes = sum(node_counts)
        forest_num_leaves = sum(leaf_counts)

        return {
            'feature_importance': feature_importance,
            'node_counts': node_counts,
            'leaf_counts': leaf_counts,
            'forest_depths': forest_depths,
            'forest_num_nodes': forest_num_nodes,
            'forest_num_leaves': forest_num_leaves
        }


if __name__ == '__main__':
    x, y = make_classification(n_classes=3, n_informative=10)
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    rf = RandomForestClassifier(n_estimators=4)

    rf.fit(x_train, y_train)

    test = PropertyTest()
    test_result = test.run_test(rf, (x_train, x_test, y_train, y_test))
    print(test_result)
