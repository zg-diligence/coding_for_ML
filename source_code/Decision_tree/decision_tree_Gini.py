import os
from PIL import Image
from uuid import uuid4
from collections import Counter, namedtuple

class DecisionTree:
    def __init__(self, data_set, features):
        self.data_set = data_set        # 数据集
        self.features = features        # 属性集合
        self.attrs = {}                 # 特征集合
        for col in range(len(data_set[0])-1):
            values = [row[col] for row in data_set]
            self.attrs[features[col]] = list(set(values))
        self.attr_num = len(self.attrs) # 特征数目

    def extractValues(self, data_set, col=-1):
        """提取每行数据指定列的值"""
        return [row[col] for row in data_set]

    def giniImpurity(self, values):
        """计算数据集的基尼系数"""
        results = {key: values.count(key) for key in set(values)}
        probs = [v/len(values) for v in results.values()]
        return sum([prob*(1-prob) for prob in probs])

    def divideNode(self, data_set, col, value):
        """分割结点"""
        split_sets = {}
        if isinstance(value, int) or isinstance(value, float):
            split_function = lambda row: row[col] >= value
        else:
            split_function = lambda row: row[col] == value
        split_sets['True'] = [row for row in data_set if split_function(row)]
        split_sets['False'] = [row for row in data_set if not split_function(row)]
        return split_sets

    def bestSplitAttr(self, data_set, values, used_attr):
        """计算最佳分割属性及属性值"""
        max_gain, split_criteria, split_sets = 0, None, None
        for col in range(self.attr_num):
            for attr in self.attrs[features[col]]:
                if attr in used_attr: continue
                sets = self.divideNode(data_set, col, attr)
                p = len(sets['True'])/ len(data_set)
                current_gini = self.giniImpurity(values)
                childs_gini = p*self.giniImpurity(self.extractValues(sets['True'])) + (1-p)*self.giniImpurity(self.extractValues(sets['False']))
                gain = current_gini - childs_gini
                if gain > max_gain and len(sets['True']) and len(sets['False']):
                    max_gain = gain
                    split_criteria = col, attr
                    split_sets = sets
        return max_gain, split_criteria, split_sets

    def buildTree(self, data_set = None, used_attr = None, threshold = 0):
        """递归建立决策树"""
        if data_set is None: data_set = self.data_set
        if used_attr is None: used_attr = []

        values = self.extractValues(data_set)
        if len(set(values)) == 1:
            return [(values[0], len(values)),]
        if len(used_attr) == self.attr_num:
            return list(Counter(values).items())
        max_gain, (split_col, attr), split_sets = self.bestSplitAttr(data_set, values, used_attr)

        if max_gain > threshold:
            feature = self.features[split_col] + ":" + str(attr)
            tree = {}; tree[feature] = {}
            for attr, subset in split_sets.items():
                tree[feature][attr] = self.buildTree(subset, used_attr + [feature,], threshold)
            return tree
        return list(Counter(values).items())

    def postprune(self, tree, threshold):
        """决策树后剪枝"""
        feature = list(tree.keys())[0]
        for attr, child in tree[feature].items():
            if isinstance(child, dict):
                tree[feature][attr] = self.postprune(child, threshold)

        # 当前结点所有子节点均为叶结点
        if not list(filter(lambda x: not isinstance(x, list), tree[feature].values())):
            left_child,right_child = [], []
            for label, count in tree[feature]['True']:
                left_child += [[label]] * count
            for label, count in tree[feature]['False']:
                right_child += [[label]] * count

            # 计算信息增益
            left_child = self.extractValues(left_child)
            right_child = self.extractValues(right_child)
            total = left_child + right_child
            p = len(left_child)/len(total)
            delta = self.giniImpurity(total) - p*self.giniImpurity(left_child) + (1-p)*self.giniImpurity(right_child)

            # 合并叶结点
            if delta >= threshold: return tree
            return list(Counter(total).items())
        return tree

    def classify(self, observation, tree):
        """对新的观测数据进行分类"""
        if not isinstance(tree, dict):  # 投票决定分类结果
            return max(dict(tree).items(), key=lambda x: x[1])[0]

        feature_attr = list(tree.keys())[0]
        feature, attr = feature_attr.split(':')
        value = observation[self.features.index(feature)]
        if isinstance(value,int) or isinstance(value, float):
            branch = tree[feature_attr]['True'] if value >= int(attr) else tree[feature_attr]['False']
        else:
            branch = tree[feature_attr]['True'] if value == attr else tree[feature_attr]['False']
        return self.classify(observation, branch)

    def getNodesEdges(self, tree, root_node=None):
        """返回决策树中所有的边和结点"""
        if not isinstance(tree, dict):
            return [], []
        Node = namedtuple('Node', ['id', 'label'])
        Edge = namedtuple('Edge', ['start', 'end', 'label'])
        nodes, edges = [], []
        if root_node is None:
            label = list(tree.keys())[0]
            root_node = Node._make([uuid4(), label])
            nodes.append(root_node)
        for edge_label, subtree in tree[root_node.label].items():
            if isinstance(subtree, dict):
                node_label = list(subtree.keys())[0]
            else:
                node_label = max(dict(subtree).items(), key=lambda x: x[1])[0]
            sub_node = Node._make([uuid4(), node_label])
            edge = Edge._make([root_node, sub_node, edge_label])
            nodes.append(sub_node)
            edges.append(edge)
            sub_nodes, sub_edges = self.getNodesEdges(subtree, sub_node)
            nodes.extend(sub_nodes)
            edges.extend(sub_edges)
        return nodes, edges

    def dotify(self, tree):
        """生成Graphviz Dot文件内容"""
        content = 'digraph decision_tree {\n'
        nodes, edges = self.getNodesEdges(tree)
        for node in nodes:
            content += '    "{}" [label="{}"];\n'.format(node.id, node.label)
        for edge in edges:
            content += '    "{}" -> "{}" [label="{}"];\n'.format(edge.start.id, edge.end.id, edge.label)
        content += '    NULL[style=invis];\n}'
        return content

if __name__ == '__main__':
    my_data = [['slashdot', 'USA', 'yes', 18, 'None'],
               ['google', 'France', 'yes', 23, 'Premium'],
               ['digg', 'USA', 'yes', 24, 'Basic'],
               ['kiwitobes', 'France', 'yes', 23, 'Basic'],
               ['google', 'UK', 'no', 21, 'Premium'],
               ['(direct)', 'New Zealand', 'no', 12, 'None'],
               ['(direct)', 'UK', 'no', 21, 'Basic'],
               ['google', 'USA', 'no', 24, 'Premium'],
               ['slashdot', 'France', 'yes', 19, 'None'],
               ['digg', 'USA', 'no', 18, 'None'],
               ['google', 'UK', 'no', 18, 'None'],
               ['kiwitobes', 'UK', 'no', 19, 'None'],
               ['digg', 'New Zealand', 'yes', 12, 'Basic'],
               ['slashdot', 'UK', 'no', 21, 'None'],
               ['google', 'UK', 'yes', 18, 'Basic'],
               ['kiwitobes', 'France', 'yes', 19, 'Basic']]

    features = ['company', 'country', 'sex', 'age']
    DT = DecisionTree(my_data, features)
    tree = DT.buildTree()
    # tree = DT.postprune(tree, 0.4)
    # print(DT.classify(['(direct)', 'UK', 'yes', 25], tree))

    with open('decision_tree.dot', 'w') as f:
        f.write(DT.dotify(tree))
    os.system('dot decision_tree.dot -T gif -o decision_tree.gif')
    Image.open('decision_tree.gif').show()