import graphviz


class Visualisation():

    def __init__(self, dt, data):
        self.dt = dt
        self.data = data

        self.dot = graphviz.Graph("decisiontree")

        self.curr_parent_node = None
        self.counter = 0
        self.knownnodes = []
        self._build_tree(self.dt.root)

    def save(self):

        self.dot.render(directory='tree').replace('\\', '/')

    def _build_tree(self, node):
        if node.value is None:
            node_name = str(self.data.columns[node.feature_index]) + " <= " + str(node.threshold) + " ? " + str(
                node.info_gain)

            # print("X_" + str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)

            if len(self.knownnodes) == 0:
                node_name = node_name + " ROOT"

            self.dot.node(node_name)
            self.knownnodes.append(node_name)
            if node.left is not None:
                left_name = str(self.data.columns[node.left.feature_index]) + " <= " + str(
                    node.left.threshold) + " ? " + str(
                    node.left.info_gain)
                self._build_tree(node.left)
                if node.left.value is None:
                    self.dot.edge(node_name, left_name, label="yes")
                else:
                    self.dot.edge(node_name, self.curr_parent_node, label="yes")

            if node.right is not None:

                right_name = str(self.data.columns[node.right.feature_index]) + " <= " + str(
                    node.right.threshold) + " ? " + str(
                    node.right.info_gain)
                self.curr_parent_node = node_name
                self._build_tree(node.right)
                if node.right.value is None:
                    self.dot.edge(node_name, right_name, label="no")
                else:
                    self.dot.edge(node_name, self.curr_parent_node, label="no")
        else:
            node_name = str(self.counter)
            self.dot.node(node_name, label=str(node.value))
            self.curr_parent_node = node_name
            self.counter += 1
