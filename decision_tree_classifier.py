import math


class Node:
    # Right child is yes branch
    # Left child is no branch
    # remaining includes the leftover data at the node
    # mvc is most valuable child
    def __init__(self, value):
        self.value = value
        self.remaining = []
        self.labels = []
        self.right_child = None
        self.left_child = None
        self.mvc_idx = None
        self.info_gain = None
    
    def get_leaf_nodes(self, max_depth):
        res = []
        if max_depth >= 0:
            if type(self.value) is str:
                res.append(self)
            else:
                res += [leaf for leaf in self.right_child.get_leaf_nodes(max_depth - 1)]
                res += [leaf for leaf in self.left_child.get_leaf_nodes(max_depth - 1)]
        return res
    
    def pretty_print(self, prefix, words, num_spaces):
        if type(self.value) is str:
            print ' '*num_spaces, prefix, self.value
        else:
            print ' '*num_spaces, prefix, words[self.value]
        print ' '*num_spaces, "Information Gain:", self.info_gain
        if self.right_child and self.left_child:
            self.right_child.pretty_print(words[self.value] + ' Node[Yes branch child]:', words, num_spaces + 2)
            self.left_child.pretty_print(words[self.value] + ' Node[No branch child]:', words, num_spaces + 2)


def get_info_gain(data, labels, word_idx):
    c1_count = labels.count('1')
    c2_count = labels.count('2')
    if c1_count == 0 and c2_count == 0:
        return 0.0
    new_ce = 0
    q = c2_count * 1.0 / (c2_count + c1_count)
    # Compute entropy
    old_ce = -1 * (q * math.log(q, 2) + (1 - q) * math.log(1 - q, 2)) if q != 0 and q != 1 else 0
    for value in [0, 1]:
        c1_new = len([x for x in range(len(data))
                      if data[x][word_idx] == value and labels[x] == '1'])
        c2_new = len([x for x in range(len(data))
                      if data[x][word_idx] == value and labels[x] == '2'])
        if c2_new != 0 or c1_new != 0:
            q = 1.0 * c2_new / (c2_new + c1_new)
            ce = -1 * (q * math.log(q, 2) + (1 - q) * math.log(1 - q, 2)) if q != 0 and q != 1 else 0
            new_ce += ((c2_new + c1_new) * 1.0 / (c2_count + c1_count)) * ce
    return old_ce - new_ce


# Generate a matrix from data
def get_matrix(words, data, labels):
    mtx = []
    for i in range(len(labels)):
        mtx.append([0] * len(words))
    for i in data:
        mtx[i[0] - 1][i[1] - 1] = 1
    return mtx


def get_node_val(idx, inp_data, labels, value):
    c1 = len([i for i in range(len(inp_data)) if inp_data[i][idx] == value and labels[i] == '1'])
    c2 = len([i for i in range(len(inp_data)) if inp_data[i][idx] == value and labels[i] == '2'])
    res = '1' if c1 > c2 else '2'
    return res


def data_loader():
    print "Data loading begin"
    with open('trainData.txt', 'r') as f:
        train_data = [[int(x) for x in line.split()] for line in f]
    with open('trainLabel.txt', 'r') as f:
        train_labels = [line.split()[0] for line in f]
    with open('testData.txt', 'r') as f:
        test_data = [[int(x) for x in line.split()] for line in f]
    with open('testLabel.txt', 'r') as f:
        test_labels = [line.split()[0] for line in f]
    with open('words.txt', 'r') as f:
        words = [line.split()[0] for line in f]
    print "Data loading Done"
    return words, train_data, train_labels, test_data, test_labels


# Decision tree classifier
def decision_tree_classifier(trn_data, trn_labels, test_data, test_labels, max_depth, max_nodes=None):
    information_gains = [get_info_gain(trn_data, trn_labels, i) for i in range(len(trn_data[0]))]
    max_gain = max(information_gains)
    mv_node = None
    try:
        mv_node = information_gains.index(max_gain)
    except ValueError:
        print "Unexpected Error"

    # Initialize root with the first maximum gain word
    root = Node(mv_node)

    rt_branch_dat = []
    rt_branch_labels = []
    lt_branch_dat = []
    lt_branch_labels = []
    for i in range(len(trn_data)):
        if trn_data[i][mv_node]:
            rt_branch_dat.append(trn_data[i])
            rt_branch_labels.append(trn_labels[i])
        else:
            lt_branch_dat.append(trn_data[i])
            lt_branch_labels.append(trn_labels[i])

    root.left_child = Node(get_node_val(mv_node, lt_branch_dat, lt_branch_labels, 0))
    root.right_child = Node(get_node_val(mv_node, rt_branch_dat, rt_branch_labels, 1))

    root.right_child.remaining = rt_branch_dat
    root.right_child.labels = rt_branch_labels
    root.left_child.remaining = lt_branch_dat
    root.left_child.labels = lt_branch_labels
    root.info_gain = max_gain
    node_count = 1
    if max_nodes is None:
        max_nodes = 2**max_depth-1
    max_nodes = min(max_nodes, 2**max_depth-1)  # restrict maximum number of nodes
    prev_acc = 0
    while node_count < max_nodes:
        if ((node_count+1) & node_count) == 0:
            print "Testing decision tree with ", node_count, "nodes"
            train_acc = test_decision_tree(root, trn_data, trn_labels)
            test_acc = test_decision_tree(root, test_data, test_labels)
            print "Train set accuracy:", train_acc
            print "Test set accuracy", test_acc
            if train_acc == prev_acc:
                break
            else:
                prev_acc = train_acc

        leaf_nodes = root.get_leaf_nodes(max_depth - 1)
        for leaf in leaf_nodes:
            if leaf.info_gain is None:
                information_gains = [get_info_gain(leaf.remaining, leaf.labels, i) for i in range(len(trn_data[0]))]
                leaf.info_gain = max(information_gains)
                leaf.mvc_idx = information_gains.index(max(information_gains))

        # Use the leaf with the highest information gain
        leaves_gain = [leaf.info_gain for leaf in leaf_nodes]
        mvc_leaf = leaf_nodes[leaves_gain.index(max(leaves_gain))]
        mvc_leaf.value = mvc_leaf.mvc_idx

        lt_branch_dat = []
        lt_branch_labels = []
        rt_branch_dat = []
        rt_branch_labels = []

        for i in range(len(mvc_leaf.remaining)):
            if mvc_leaf.remaining[i][mvc_leaf.mvc_idx]:
                rt_branch_dat.append(mvc_leaf.remaining[i])
                rt_branch_labels.append(mvc_leaf.labels[i])
            else:
                lt_branch_dat.append(mvc_leaf.remaining[i])
                lt_branch_labels.append(mvc_leaf.labels[i])
        node_count += 1
        mvc_leaf.left_child = Node(get_node_val(mvc_leaf.mvc_idx, mvc_leaf.remaining, mvc_leaf.labels, 0))
        mvc_leaf.right_child = Node(get_node_val(mvc_leaf.mvc_idx, mvc_leaf.remaining, mvc_leaf.labels, 1))
        mvc_leaf.right_child.remaining = rt_branch_dat
        mvc_leaf.right_child.labels = rt_branch_labels
        mvc_leaf.left_child.remaining = lt_branch_dat
        mvc_leaf.left_child.labels = lt_branch_labels

    return root


# Return accuracy
def test_decision_tree(tree, input_matrix, labels):
    def infer(root, doc):
        node = root
        while not (type(node.value) is str):
            if doc[node.value] == 1:
                node = node.right_child
            else:
                node = node.left_child
        return node.value
    results = [infer(tree, item) for item in input_matrix]
    correct = len([i for i in range(len(labels)) if labels[i] == results[i]])
    return correct*1.0 / len(labels)


# Change max depth and max nodes in the decision_tree_learner call for new experiments
# Current setup is at depth=4 nodes=10
def main(max_depth=4, max_nodes=10):
    words, trn_data, trn_labels, tst_data, tst_labels = data_loader()

    trn_mtx = get_matrix(words, trn_data, trn_labels)
    tst_mtx = get_matrix(words, tst_data, tst_labels)

    # for max_depth in range(20):
    #     print "Depth: ", max_depth
    #     tree = decision_tree_learner(trn_mtx, trn_labels, tst_mtx, tst_labels, max_depth)
    #     print "Train Accuracy:", test_decision_tree(tree, trn_mtx, trn_labels)
    #     print "Test Accuracy:", test_decision_tree(tree, tst_mtx, tst_labels)
    # print "Depth: ", max_depth
    tree = decision_tree_classifier(trn_mtx, trn_labels, tst_mtx, tst_labels, max_depth, max_nodes)
    print "Train Accuracy:", test_decision_tree(tree, trn_mtx, trn_labels)
    print "Test Accuracy:", test_decision_tree(tree, tst_mtx, tst_labels)

    tree.pretty_print('', words, 0)

if __name__ == '__main__':
    main()
