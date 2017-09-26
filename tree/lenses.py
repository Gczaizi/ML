# coding: utf-8

import tree
import treePlotter


if __name__ == '__main__':
    fr = open('/mnt/e/Study/data/lenses.txt')
    lenses = [line.strip().split('\t') for line in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = tree.createTree(lenses, lensesLabels)
    treePlotter.createPlot(lensesTree)