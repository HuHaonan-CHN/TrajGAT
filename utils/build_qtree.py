import collections
from tqdm.std import tqdm
import sys

sys.path.append("/home/huhaonan/nfs/huhaonan/TrajectoryRepresentation/GraphTransformer")


from utils.tools import pdump, pload
from utils.qtree import Index


def build_qtree(traj_data, x_range, y_range, max_items, max_depth):
    qtree = Index(bbox=(x_range[0], y_range[0], x_range[1], y_range[1]), max_items=max_items, max_depth=max_depth)
    point_num = 0

    for traj in tqdm(traj_data):
        for point in traj:
            point_num += 1
            x, y = point
            qtree.insert(point_num, (x, y, x, y))

    print("traj point nums:", point_num)

    return qtree


if __name__ == "__main__":
    # # traj_data = pload("/home/huhaonan/nfs/huhaonan/Data/porto_back/nor/porto_mix/val_mix_trajs_9958_all.pkl")  # Mix
    # traj_data = pload("/home/huhaonan/nfs/huhaonan/Data/porto_back/nor/porto_long/val_long_trajs_10000_all.pkl")  # Long
    # qtree = build_qtree(traj_data, [-15.630759, -3.930948], [36.886104, 45.657225], max_items=50, max_depth=50)

    # traj_data = pload("/home/huhaonan/nfs/huhaonan/Data/DiDi/mix/traj/mix_trajs_10000.pkl")
    traj_data = pload("/home/huhaonan/nfs/huhaonan/Data/DiDi/long/traj/long_trajs_10000.pkl")  # Long

    qtree = build_qtree(traj_data, [108.91114, 108.9986], [34.2053, 34.28022], max_items=50, max_depth=20)

    # print("The depth of Q-Tree:", qtree._depth)
    print(qtree.nodes)
    print(qtree.children)
    queue = collections.deque([qtree])
    max_depth = -1

    while queue:
        range_length = len(queue)
        max_depth += 1
        for _ in range(range_length):
            node = queue.popleft()
            for c in node.children:
                queue.append(c)
    print(max_depth)

    vir_node_num = 0
    vir_node_dict = {}
    for vir_node in qtree:
        # print(vir_node)
        # if len(vir_node.nodes) > 0 or len(vir_node.children)>0:
        if len(vir_node.nodes) > 0:
            vir_node_num += 1
            depth = vir_node._depth
            if depth in vir_node_dict.keys():
                vir_node_dict[depth].append(vir_node)
            else:
                vir_node_dict[depth] = [vir_node]
    print("layers \t vir_n_num \t mean_node \t max_node \t min_node")
    for layers in vir_node_dict.keys():
        print("{} \t {} \t {:.2f} \t {} \t {}".format(layers, len(vir_node_dict[layers]), sum([len(n.nodes) for n in vir_node_dict[layers]]) / len(vir_node_dict[layers]), max([len(n.nodes) for n in vir_node_dict[layers]]), min([len(n.nodes) for n in vir_node_dict[layers]]),))
    print(vir_node_num)
    # get_rtree_structure(qtree)

