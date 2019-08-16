
from __future__ import division
import numpy as np

from base import BaseEstimator, ClassifierMixin, RegressorMixin
from selector_mixin import SelectorMixin

from . import _tree

def array2d(X, dtype=None, order=None):
    """Returns at least 2-d array with data from X"""
    return np.asarray(np.atleast_2d(X), dtype=dtype, order=order)

def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

__all__ = ["DecisionTreeClassifier",
           "DecisionTreeRegressor",
           "ExtraTreeClassifier",
           "ExtraTreeRegressor"]

DTYPE = _tree.DTYPE

CLASSIFICATION = {
    "gini": _tree.Gini,
    "entropy": _tree.Entropy,
}

REGRESSION = {
    "mse": _tree.MSE,
}

GRAPHVIZ_TREE_TEMPLATE = """\
%(current)s [label="%(current_gv)s"] ;
%(left_child)s [label="%(left_child_gv)s"] ;
%(right_child)s [label="%(right_child_gv)s"] ;
%(current)s -> %(left_child)s ;
%(current)s -> %(right_child)s ;
"""


def export_graphviz(decision_tree, out_file=None, feature_names=None):
    def node_to_str(tree, node_id):
        if tree.children[node_id, 0] == tree.children[node_id, 1] == Tree.LEAF:
            return "error = %s\\nsamples = %s\\nvalue = %s" \
                % (tree.init_error[node_id], tree.n_samples[node_id],
                   tree.value[node_id])
        else:
            if feature_names is not None:
                feature = feature_names[tree.feature[node_id]]
            else:
                feature = "X[%s]" % tree.feature[node_id]

            return "%s <= %s\\nerror = %s\\nsamples = %s\\nvalue = %s" \
                   % (feature, tree.threshold[node_id],
                      tree.init_error[node_id], tree.n_samples[node_id],
                      tree.value[node_id])

    def recurse(tree, node_id):
        assert node_id != -1
        left_child, right_child = tree.children[node_id, :]
        node_data = {
            "current": node_id,
            "current_gv": node_to_str(tree, node_id),
            "left_child": left_child,
            "left_child_gv": node_to_str(tree, left_child),
            "right_child": right_child,
            "right_child_gv": node_to_str(tree, right_child),
        }

        out_file.write(GRAPHVIZ_TREE_TEMPLATE % node_data)

        if not (tree.children[left_child, 0] == tree.children[left_child, 1] \
                == Tree.LEAF):
            recurse(tree, left_child)
        if not (tree.children[right_child, 0] == \
                tree.children[right_child, 1] == Tree.LEAF):
            recurse(tree, right_child)

    if out_file is None:
        out_file = open("tree.dot", "w")
    elif isinstance(out_file, basestring):
        out_file = open(out_file, "w")

    out_file.write("digraph Tree {\n")
    recurse(decision_tree.tree_, 0)
    out_file.write("}")

    return out_file


class Tree(object):

    LEAF = -1
    UNDEFINED = -2

    def __init__(self, k, capacity=3):
        self.node_count = 0

        self.children = np.empty((capacity, 2), dtype=np.int32)
        self.children.fill(Tree.UNDEFINED)

        self.feature = np.empty((capacity,), dtype=np.int32)
        self.feature.fill(Tree.UNDEFINED)

        self.threshold = np.empty((capacity,), dtype=np.float64)
        self.value = np.empty((capacity, k), dtype=np.float64)

        self.best_error = np.empty((capacity,), dtype=np.float32)
        self.init_error = np.empty((capacity,), dtype=np.float32)
        self.n_samples = np.empty((capacity,), dtype=np.int32)

    def resize(self, capacity=None):
        """Resize tree arrays to `capacity`, if `None` double capacity. """
        if capacity is None:
            capacity = int(self.children.shape[0] * 2.0)

        if capacity == self.children.shape[0]:
            return

        self.children.resize((capacity, 2), refcheck=False)
        self.feature.resize((capacity,), refcheck=False)
        self.threshold.resize((capacity,), refcheck=False)
        self.value.resize((capacity, self.value.shape[1]), refcheck=False)
        self.best_error.resize((capacity,), refcheck=False)
        self.init_error.resize((capacity,), refcheck=False)
        self.n_samples.resize((capacity,), refcheck=False)

        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

    def add_split_node(self, parent, is_left_child, feature, threshold,
                       best_error, init_error, n_samples, value):
        """Add a splitting node to the tree. The new node registers itself as
        the child of its parent. """
        node_id = self.node_count
        if node_id >= self.children.shape[0]:
            self.resize()

        self.feature[node_id] = feature
        self.threshold[node_id] = threshold

        self.init_error[node_id] = init_error
        self.best_error[node_id] = best_error
        self.n_samples[node_id] = n_samples
        self.value[node_id] = value

        # set as left or right child of parent
        if parent > Tree.LEAF:
            if is_left_child:
                self.children[parent, 0] = node_id
            else:
                self.children[parent, 1] = node_id

        self.node_count += 1
        return node_id

    def add_leaf(self, parent, is_left_child, value, error, n_samples):
        """Add a leaf to the tree. The new node registers itself as the
        child of its parent. """
        node_id = self.node_count
        if node_id >= self.children.shape[0]:
            self.resize()

        self.value[node_id] = value
        self.n_samples[node_id] = n_samples
        self.init_error[node_id] = error
        self.best_error[node_id] = error

        if is_left_child:
            self.children[parent, 0] = node_id
        else:
            self.children[parent, 1] = node_id

        self.children[node_id, :] = Tree.LEAF

        self.node_count += 1

    def predict(self, X):
        out = np.empty((X.shape[0], ), dtype=np.int32)
        _tree._apply_tree(X, self.children, self.feature, self.threshold, out)
        return self.value.take(out, axis=0)


def _build_tree(X, y, is_classification, criterion, max_depth, min_split,
                min_density, max_features, random_state, n_classes, find_split,
                sample_mask=None, X_argsorted=None):
    """Build a tree by recursively partitioning the data."""

    if max_depth <= 10:
        init_capacity = (2 ** (max_depth + 1)) - 1
    else:
        init_capacity = 2047  # num nodes of tree with depth 10

    tree = Tree(n_classes, init_capacity)

    # Recursively partition X
    def recursive_partition(X, X_argsorted, y, sample_mask, depth,
                            parent, is_left_child):
        # Count samples
        n_node_samples = sample_mask.sum()

        if n_node_samples == 0:
            raise ValueError("Attempting to find a split "
                             "with an empty sample_mask")

        # Split samples
        if depth < max_depth and n_node_samples >= min_split:
            feature, threshold, best_error, init_error = find_split(
                X, y, X_argsorted, sample_mask, n_node_samples,
                max_features, criterion, random_state)

        else:
            feature = -1

        # Value at this node
        current_y = y[sample_mask]

        if is_classification:
            value = np.zeros((n_classes,))
            t = current_y.max() + 1
            value[:t] = np.bincount(current_y.astype(np.int))

        else:
            value = np.asarray(np.mean(current_y))

        # Terminal node
        if feature == -1:
            # compute error at leaf
            error = _tree._error_at_leaf(y, sample_mask, criterion,
                                         n_node_samples)
            tree.add_leaf(parent, is_left_child, value, error, n_node_samples)

        # Internal node
        else:
            # Sample mask is too sparse?
            if n_node_samples / X.shape[0] <= min_density:
                X = X[sample_mask]
                X_argsorted = np.asfortranarray(
                    np.argsort(X.T, axis=1).astype(np.int32).T)
                y = current_y
                sample_mask = np.ones((X.shape[0],), dtype=np.bool)

            # Split and and recurse
            split = X[:, feature] <= threshold

            node_id = tree.add_split_node(parent, is_left_child, feature,
                                          threshold, best_error, init_error,
                                          n_node_samples, value)

            # left child recursion
            recursive_partition(X, X_argsorted, y,
                                split & sample_mask,
                                depth + 1, node_id, True)

            # right child recursion
            recursive_partition(X, X_argsorted, y,
                                ~split & sample_mask,
                                depth + 1, node_id, False)

    # Launch the construction
    if X.dtype != DTYPE or not np.isfortran(X):
        X = np.asanyarray(X, dtype=DTYPE, order="F")

    if y.dtype != DTYPE or not y.flags.contiguous:
        y = np.ascontiguousarray(y, dtype=DTYPE)

    if sample_mask is None:
        sample_mask = np.ones((X.shape[0],), dtype=np.bool)

    if X_argsorted is None:
        X_argsorted = np.asfortranarray(
            np.argsort(X.T, axis=1).astype(np.int32).T)

    recursive_partition(X, X_argsorted, y, sample_mask, 0, -1, False)
    tree.resize(tree.node_count)

    return tree


class BaseDecisionTree(BaseEstimator, SelectorMixin):
    def __init__(self, criterion,
                       max_depth,
                       min_split,
                       min_density,
                       max_features,
                       compute_importances,
                       random_state):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_split = min_split
        self.min_density = min_density
        self.max_features = max_features
        self.compute_importances = compute_importances
        self.random_state = check_random_state(random_state)

        self.n_features_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.find_split_ = _tree._find_best_split

        self.tree_ = None
        self.feature_importances_ = None

    def fit(self, X, y, sample_mask=None, X_argsorted=None):
        # Convert data
        X = np.asarray(X, dtype=DTYPE, order='F')
        n_samples, self.n_features_ = X.shape

        is_classification = isinstance(self, ClassifierMixin)

        if is_classification:
            self.classes_ = np.unique(y)
            self.n_classes_ = self.classes_.shape[0]
            criterion = CLASSIFICATION[self.criterion](self.n_classes_)
            y = np.searchsorted(self.classes_, y)

        else:
            self.classes_ = None
            self.n_classes_ = 1
            criterion = REGRESSION[self.criterion]()

        y = np.ascontiguousarray(y, dtype=DTYPE)

        # Check parameters
        max_depth = np.inf if self.max_depth is None else self.max_depth

        if isinstance(self.max_features, basestring):
            if self.max_features == "auto":
                if is_classification:
                    max_features = max(1, int(np.sqrt(self.n_features_)))

                else:
                    max_features = self.n_features_

            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))

            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))

            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')

        elif self.max_features is None:
            max_features = self.n_features_

        else:
            max_features = self.max_features

        if len(y) != n_samples:
            raise ValueError("Number of labels=%d does not match "
                             "number of samples=%d" % (len(y), n_samples))
        if self.min_split <= 0:
            raise ValueError("min_split must be greater than zero.")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")
        if self.min_density < 0.0 or self.min_density > 1.0:
            raise ValueError("min_density must be in [0, 1]")
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")

        # Build tree
        self.tree_ = _build_tree(X, y, is_classification, criterion,
                                max_depth, self.min_split,
                                self.min_density, max_features,
                                self.random_state, self.n_classes_,
                                self.find_split_, sample_mask=sample_mask,
                                X_argsorted=X_argsorted)

        # Compute feature importances
        if self.compute_importances:
            importances = np.zeros(self.n_features_)

            for node in xrange(self.tree_.node_count):
                if (self.tree_.children[node, 0]
                        == self.tree_.children[node, 1]
                        == Tree.LEAF):
                    continue

                else:
                    importances[self.tree_.feature[node]] += (
                        self.tree_.n_samples[node] *
                            (self.tree_.init_error[node] -
                             self.tree_.best_error[node]))

            importances /= np.sum(importances)
            self.feature_importances_ = importances

        return self

    def predict(self, X):
        X = array2d(X, dtype=DTYPE)
        n_samples, n_features = X.shape

        if self.tree_ is None:
            raise Exception("Tree not initialized. Perform a fit first")

        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             " match the input. Model n_features is %s and "
                             " input n_features is %s "
                             % (self.n_features_, n_features))

        if isinstance(self, ClassifierMixin):
            predictions = self.classes_.take(np.argmax(
                self.tree_.predict(X), axis=1), axis=0)
        else:
            predictions = self.tree_.predict(X).ravel()

        return predictions


class DecisionTreeClassifier(BaseDecisionTree, ClassifierMixin):
    def __init__(self, criterion="gini",
                       max_depth=None,
                       min_split=1,
                       min_density=0.1,
                       max_features=None,
                       compute_importances=False,
                       random_state=None):
        super(DecisionTreeClassifier, self).__init__(criterion,
                                                     max_depth,
                                                     min_split,
                                                     min_density,
                                                     max_features,
                                                     compute_importances,
                                                     random_state)

    def predict_proba(self, X):
        X = array2d(X, dtype=DTYPE)
        n_samples, n_features = X.shape

        if self.tree_ is None:
            raise Exception("Tree not initialized. Perform a fit first.")

        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             " match the input. Model n_features is %s and "
                             " input n_features is %s "
                             % (self.n_features_, n_features))

        P = self.tree_.predict(X)
        P /= P.sum(axis=1)[:, np.newaxis]
        return P

    def predict_log_proba(self, X):
        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class log-probabilities of the input samples. Classes are
            ordered by arithmetical order.
        return np.log(self.predict_proba(X))


class DecisionTreeRegressor(BaseDecisionTree, RegressorMixin):
    def __init__(self, criterion="mse",
                       max_depth=None,
                       min_split=1,
                       min_density=0.1,
                       max_features=None,
                       compute_importances=False,
                       random_state=None):
        super(DecisionTreeRegressor, self).__init__(criterion,
                                                    max_depth,
                                                    min_split,
                                                    min_density,
                                                    max_features,
                                                    compute_importances,
                                                    random_state)


class ExtraTreeClassifier(DecisionTreeClassifier):
    def __init__(self, criterion="gini",
                       max_depth=None,
                       min_split=1,
                       min_density=0.1,
                       max_features="auto",
                       compute_importances=False,
                       random_state=None):
        super(ExtraTreeClassifier, self).__init__(criterion,
                                                  max_depth,
                                                  min_split,
                                                  min_density,
                                                  max_features,
                                                  compute_importances,
                                                  random_state)

        self.find_split_ = _tree._find_best_random_split


class ExtraTreeRegressor(DecisionTreeRegressor):
    def __init__(self, criterion="mse",
                       max_depth=None,
                       min_split=1,
                       min_density=0.1,
                       max_features="auto",
                       compute_importances=False,
                       random_state=None):
        super(ExtraTreeRegressor, self).__init__(criterion,
                                                 max_depth,
                                                 min_split,
                                                 min_density,
                                                 max_features,
                                                 compute_importances,
                                                 random_state)

        self.find_split_ = _tree._find_best_random_split
